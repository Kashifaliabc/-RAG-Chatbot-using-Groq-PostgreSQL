import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings  # Using HuggingFace instead of OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PG_CONN_STRING = os.getenv("connection_string")
# OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")  # No longer needed

# Streamlit App Title
st.title("🧠 RAG Chatbot using Groq + PostgreSQL")

# File Upload
uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Save and process the uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embed and store in PostgreSQL
    st.info("Storing document in vector database...")

    # Use HuggingFace embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_docs",
        connection_string=PG_CONN_STRING
    )
    st.success("Document stored successfully!")

# Initialize memory once
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# User Query
query = st.text_input("💬 Ask a question based on your uploaded docs:")

if query:
    # Initialize Groq LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

    # Initialize retriever using the PGVector store
    retriever = PGVector(
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        collection_name="rag_docs",
        connection_string=PG_CONN_STRING
    ).as_retriever()

    # Create a ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        verbose=False
    )

    # Get the response
    response = qa_chain.run(query)
    st.write("🤖", response)

# Sidebar Chat History and Clear Option
with st.sidebar:
    st.markdown("## 🧠 Chat History")
    if st.session_state.memory.chat_memory.messages:
        for msg in st.session_state.memory.chat_memory.messages:
            role = "🧑‍💻 You" if msg.type == "human" else "🤖 Bot"
            st.markdown(f"**{role}:** {msg.content}")

        if st.button("🧹 Clear Chat History"):
            st.session_state.memory.clear()
            st.experimental_rerun()
    else:
        st.info("Start chatting to see history here.")
