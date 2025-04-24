# 🧠 RAG Chatbot using Groq, PostgreSQL, and LangChain

A **Retrieval-Augmented Generation (RAG)** chatbot built with **Groq LLMs**, **PostgreSQL with pgvector**, and **LangChain**. This Streamlit-based app allows users to upload PDF documents, store them as vector embeddings, and ask questions with real-time context-aware responses.

---

## 🚀 Features

- 📄 **PDF Upload** – Upload and process any PDF document.
- 🧩 **Text Chunking & Embedding** – Uses LangChain's splitter and HuggingFace's sentence-transformers.
- 🗃️ **Vector Store** – Stores embeddings in a PostgreSQL database using `pgvector`.
- 🤖 **Conversational QA** – Ask questions and get intelligent answers based on the document.
- 💬 **Chat Memory** – Remembers previous queries to maintain a contextual conversation.

---

## 🔍 How It Works

1. ### 📤 File Upload  
   Users upload a PDF file which is saved locally for processing.

2. ### ✂️ Text Chunking  
   The document is split into smaller text chunks using `RecursiveCharacterTextSplitter` to make it suitable for embedding.

3. ### 🧠 Embedding  
   Each text chunk is converted into a dense vector using the HuggingFace model `sentence-transformers/all-MiniLM-L6-v2`.

4. ### 🗂️ Vector Storage  
   The embeddings are stored in a PostgreSQL database using the `pgvector` extension, allowing for fast similarity search.

5. ### 💬 Conversational Querying  
   Users type questions, and the chatbot retrieves relevant chunks and generates answers using the Groq LLM (`llama3-8b-8192`) with chat memory.

---

## ⚙️ Setup Guide

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install Dependancy
pip install -r requirements.txt
