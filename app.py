import os
import tempfile
import time
import fitz  # PyMuPDF
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

# --- Constants ---
MAX_UPLOAD_SIZE_MB = 10  # Max file size in MB
MAX_REQUESTS_PER_MINUTE = 10  # Rate limiting
SESSION_STATE = st.session_state

# --- Rate Limiting ---
def check_rate_limit(username: str) -> bool:
    """Check if user has exceeded rate limit"""
    if 'rate_limits' not in SESSION_STATE:
        SESSION_STATE.rate_limits = {}
    
    now = datetime.now()
    if username not in SESSION_STATE.rate_limits:
        SESSION_STATE.rate_limits[username] = {
            'count': 1,
            'window_start': now
        }
        return True
    
    time_diff = (now - SESSION_STATE.rate_limits[username]['window_start']).total_seconds()
    
    if time_diff > 60:  # Reset counter if window has passed
        SESSION_STATE.rate_limits[username] = {
            'count': 1,
            'window_start': now
        }
        return True
    
    if SESSION_STATE.rate_limits[username]['count'] >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    SESSION_STATE.rate_limits[username]['count'] += 1
    return True

# --- File Processing ---
def display_pdf_viewer(pdf_path: str, page_num: int = 0, doc_name: str = None) -> Optional[fitz.Page]:
    """Display a PDF in the sidebar with page navigation
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Current page number (0-based)
        doc_name: Name of the document (used as key in current_pages)
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Display the image with responsive width
        st.sidebar.image(img, width='stretch')
        
        # Page navigation
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        
        # Get the document name for state management
        doc_key = doc_name or os.path.basename(pdf_path)
        
        with col1:
            if st.button("⏮ Previous", key=f"prev_{doc_key}", disabled=page_num <= 0):
                if doc_key in SESSION_STATE.current_pages:
                    SESSION_STATE.current_pages[doc_key] = max(0, page_num - 1)
                st.rerun()
        
        with col2:
            # Add a number input for direct page navigation
            new_page = st.number_input(
                f"Page {page_num + 1} of {len(doc)}", 
                min_value=1, 
                max_value=len(doc), 
                value=page_num + 1,
                key=f"page_input_{doc_key}",
                on_change=lambda: setattr(SESSION_STATE.current_pages, doc_key, st.session_state[f"page_input_{doc_key}"] - 1)
            )
        
        with col3:
            if st.button("Next ⏭", key=f"next_{doc_key}", disabled=page_num >= len(doc) - 1):
                if doc_key in SESSION_STATE.current_pages:
                    SESSION_STATE.current_pages[doc_key] = min(len(doc) - 1, page_num + 1)
                st.rerun()
        
        doc.close()
        return page
    except Exception as e:
        st.sidebar.error(f"Error displaying PDF: {str(e)}")
        return None

def process_pdfs(uploaded_files: List[Any]) -> Tuple[Optional[Chroma], Dict[str, str]]:
    """Process multiple PDF files and return a vector store and file paths"""
    all_docs = []
    file_paths = {}
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            file_paths[uploaded_file.name] = tmp_path
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Add metadata with file name and page number
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    'source': uploaded_file.name,
                    'page': i + 1
                })
            
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if not all_docs:
        st.error("No valid PDFs were processed.")
        return None, {}
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings), file_paths

# --- Chat History Management ---
def get_chat_history() -> List[Dict[str, str]]:
    """Get or initialize chat history"""
    if 'chat_history' not in SESSION_STATE:
        SESSION_STATE.chat_history = []
    return SESSION_STATE.chat_history

def add_to_chat_history(role: str, content: Dict[str, Any]):
    """Add a message to chat history"""
    history = get_chat_history()
    history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    # Keep only the last 50 messages to prevent memory issues
    SESSION_STATE.chat_history = history[-50:]

# --- Main Application ---
def main():
    st.set_page_config(page_title="DocuMind", page_icon="", layout="wide")
    
    # Initialize session state
    if 'username' not in SESSION_STATE:
        SESSION_STATE.username = "default_user"
    if 'current_pages' not in SESSION_STATE:
        SESSION_STATE.current_pages = {}
    if 'last_query' not in SESSION_STATE:
        SESSION_STATE.last_query = ""
    
    st.title("DocuMind: Chat with your PDFs")
    
    # --- Main Layout ---
    col1, col2 = st.columns([3, 1])
    
    with col2:  # Right sidebar for document viewing
        st.sidebar.header("Document Viewer")
        
        # Document uploader
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDFs", 
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Max file size: {MAX_UPLOAD_SIZE_MB}MB per file"
        )
        
        if uploaded_files:
            # Check file sizes
            valid_files = []
            for file in uploaded_files:
                if file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    st.sidebar.warning(f"File {file.name} exceeds {MAX_UPLOAD_SIZE_MB}MB and will be skipped.")
                else:
                    valid_files.append(file)
            
            if valid_files:
                with st.spinner("Processing documents..."):
                    try:
                        SESSION_STATE.vectordb, SESSION_STATE.file_paths = process_pdfs(valid_files)
                        if SESSION_STATE.vectordb:
                            SESSION_STATE.retriever = SESSION_STATE.vectordb.as_retriever()
                            st.sidebar.success(f"Processed {len(valid_files)} file(s) successfully!")
                            
                            # Initialize current page for each document
                            for file_name in SESSION_STATE.file_paths:
                                if file_name not in SESSION_STATE.current_pages:
                                    SESSION_STATE.current_pages[file_name] = 0
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        # Document selection and display
        if hasattr(SESSION_STATE, 'file_paths') and SESSION_STATE.file_paths:
            selected_doc = st.sidebar.selectbox(
                "Select Document",
                options=list(SESSION_STATE.file_paths.keys()),
                key="doc_selector"
            )
            
            if selected_doc:
                pdf_path = SESSION_STATE.file_paths[selected_doc]
                current_page = SESSION_STATE.current_pages.get(selected_doc, 0)
                
                # Display the PDF viewer
                display_pdf_viewer(pdf_path, current_page, selected_doc)
    
    with col1:  # Main chat area
        # Initialize LLM only if API key is provided
        if 'llm' not in SESSION_STATE:
            groq_api_key = st.sidebar.text_input(
                "Enter your Groq API Key (get it from https://console.groq.com/keys)",
                type="password"
            )
            
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
                SESSION_STATE.llm = ChatGroq(
                    temperature=0,
                    model_name="llama-3.3-70b-versatile",
                    api_key=groq_api_key
                )
            else:
                st.sidebar.warning("Please enter your Groq API key to continue")
                return
    
    # Initialize chat history
    chat_history = get_chat_history()
    
    # Reset chat history if files change
    if 'last_uploaded_files' not in SESSION_STATE:
        SESSION_STATE.last_uploaded_files = set()
    
    current_files = {f.name: f.size for f in uploaded_files} if uploaded_files else {}
    if current_files != SESSION_STATE.last_uploaded_files:
        SESSION_STATE.chat_history = []
        SESSION_STATE.last_uploaded_files = current_files
        chat_history = get_chat_history()
    
    # Display chat history
    for message in chat_history:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            # Check if content is a string or a dict with content
            content = message.get("content", "")
            if isinstance(content, dict):
                content = content.get("content", "")
            st.markdown(content)
            
            # Show source documents for assistant messages
            if role == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    sources = message["sources"]
                    for i, source in enumerate(sources[:3]):  # Show top 3 sources
                        if isinstance(source, dict):
                            page = source.get('metadata', {}).get('page', 'N/A')
                            source_content = source.get('content', '')
                        else:
                            page = getattr(source, 'metadata', {}).get('page', 'N/A')
                            source_content = getattr(source, 'page_content', '')
                        
                        st.markdown(f"**Source {i+1}** (Page {page}):")
                        st.markdown(f"> {source_content[:300]}{'...' if len(source_content) > 300 else ''}")
    
    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Store the user's question
        # Check rate limit
        if not check_rate_limit("default_user"):
            st.error("Rate limit exceeded. Please wait a minute before sending more messages.")
            st.stop()
        
        # Add user message to chat
        add_to_chat_history("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if documents are loaded
        if 'retriever' not in SESSION_STATE:
            with st.chat_message("assistant"):
                st.error("Please upload and process documents first.")
                add_to_chat_history("assistant", "Error: No documents loaded.")
            st.stop()
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build the chain
                    prompt_template = ChatPromptTemplate.from_template("""
                    You are a helpful assistant. Use ONLY the provided context.
                    If the answer is not in the document, respond: "I cannot find that information."

                    Chat History:
                    {chat_history}

                    Question: {question}

                    Context:
                    {context}
                    """)
                    
                    # Get relevant context with sources
                    docs = SESSION_STATE.retriever.invoke(prompt)
                    context = "\n\n".join([f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" 
                                      for i, doc in enumerate(docs)])
                    
                    # Format chat history for context (only user and assistant messages)
                    history_messages = []
                    for msg in chat_history[-5:]:  # Last 5 messages for context
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            if isinstance(content, dict):
                                content = content.get('content', '')
                            if role and content:
                                history_messages.append(f"{role}: {content}")
                    
                    history_str = "\n".join(history_messages)
                    
                    try:
                        # Format the prompt with proper markdown
                        formatted_prompt = f"""You are a helpful assistant. Use ONLY the provided context.
                        If the answer is not in the document, respond: "I cannot find that information."

                        Chat History:
                        {history_str}

                        Question: {prompt}

                        Context:
                        {context}

                        Please format your response with clear paragraphs and proper spacing for better readability."""
                        
                        # Generate response using the LLM
                        response = SESSION_STATE.llm.invoke(formatted_prompt)
                        
                        # Get the response content and clean it up
                        response_content = response.content if hasattr(response, 'content') else str(response)
                        response_content = response_content.strip()
                        
                        # Display the response with better formatting
                        st.markdown("### Answer")
                        st.markdown(response_content)
                        
                        # Display sources if available
                        if docs:
                            with st.expander("View Sources"):
                                for i, doc in enumerate(docs[:3]):  # Show top 3 sources
                                    st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                                    st.markdown(f"> {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
                        
                        # Store the response in chat history with clean format
                        add_to_chat_history("assistant", {
                            "content": response_content,
                            "sources": [{
                                'content': d.page_content,
                                'metadata': {
                                    'page': d.metadata.get('page', 'N/A'),
                                    'source': d.metadata.get('source', 'Unknown')
                                }
                            } for d in docs[:3]]  # Store top 3 sources
                        })
                        
                        # Don't rerun here - it causes the page to refresh and lose formatting
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        add_to_chat_history("assistant", error_msg)
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    add_to_chat_history("assistant", error_msg)

if __name__ == "__main__":
    main()