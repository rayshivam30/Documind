import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# UPDATED embedding import
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

# --- PAGE SETUP ---
st.set_page_config(page_title="DocuMind", page_icon="")
st.title("DocuMind: Chat with your PDF")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.markdown("---")

# --- MAIN ---
if uploaded_file and api_key:

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.write("📄 Processing PDF...")

    # 1. Load PDF pages
    docs = PyPDFLoader(tmp_path).load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 3. Embeddings (UPDATED)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Chroma Vector DB
    vectordb = Chroma.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever()

    # 5. Groq LLM
    llm = ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"  # Using available model
    )

    # 6. Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the provided context.
    If the answer is not in the document, respond: "I cannot find that information."

    Question: {question}

    Context:
    {context}
    """)

    # 7. Retrieval + LLM Chain (LCEL)
    chain = (
        RunnableMap({
            "context": lambda q: retriever.invoke(q),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    st.success("🎉 Ready! Ask your question.")
    st.divider()

    # --- Question Input ---
    query = st.text_input("Ask something about your PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = chain.invoke(query)
            st.markdown("### 🤖 Answer:")
            st.write(response)

    os.remove(tmp_path)

elif not api_key:
    st.warning("Enter your Groq API Key in the sidebar.")
