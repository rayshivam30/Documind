# DocuMind

DocuMind is an intelligent document assistant that allows you to chat with your PDF documents. Built with Streamlit and powered by LangChain and Groq's LLM, it enables natural language interactions with your documents, extracting relevant information through semantic search and question-answering capabilities.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Start the application with:
```
python app.py
```

## Features

- **Document Processing**: Upload and process PDF documents
- **Semantic Search**: Find relevant information using vector embeddings
- **Chat Interface**: Natural language Q&A with your documents
- **Local Processing**: Your documents stay on your machine
- **Fast Inference**: Powered by Groq's high-performance LLM

## Project Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Project dependencies
- `README.md` - This documentation

## How It Works

1. Upload a PDF document through the web interface
2. The document is processed and split into chunks
3. Chunks are converted to vector embeddings using HuggingFace models
4. Your questions are matched with relevant document sections
5. The LLM generates accurate answers based on the document content
