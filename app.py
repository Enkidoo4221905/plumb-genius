import streamlit as st
import os
import pdfplumber
import json
import csv
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
VECTOR_DIR = "./vector_store"
DATA_DIR = "./data"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize OpenAI models
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

@st.cache_resource
def load_base_knowledge() -> List[Document]:
    """Load the pre-existing knowledge base from JSON and CSV files"""
    documents = []

    # Load JSON files
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Adjust this based on your JSON structure
            text = json.dumps(data, indent=2)
            documents.extend(SPLITTER.create_documents([text]))

    # Load CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Adjust this based on your CSV structure
                text = "\n".join(f"{k}: {v}" for k, v in row.items())
                documents.extend(SPLITTER.create_documents([text]))

    return documents

def process_pdf(pdf_file) -> List[Document]:
    """Process new PDF uploads"""
    docs = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.extend(SPLITTER.create_documents([text], metadatas=[{"source": f"{pdf_file.name}_page_{i+1}"}]))
    return docs

@st.cache_resource
def initialize_vectorstore():
    """Initialize or load the vector store with base knowledge"""
    if os.path.exists(VECTOR_DIR):
        try:
            return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")

    # If vector store doesn't exist, create it from base knowledge
    base_docs = load_base_knowledge()
    if base_docs:
        vectorstore = FAISS.from_documents(base_docs, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        return vectorstore
    return None

def ask_llm(vectorstore, question: str, k: int = 5) -> str:
    try:
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"You are a plumbing expert with knowledge of the 2021 IPC.\nUse the following context to answer accurately:\n{context}\nQuestion: {question}\nAnswer:"
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        st.error(f"LLM query failed: {e}")
        return "Sorry, I couldn't process your question."

# Streamlit UI
st.title("Plumb Genius: Plumbing Code Chatbot")

# Initialize vector store with base knowledge
vectorstore = initialize_vectorstore()

# Optional PDF upload
st.subheader("Upload Additional Documents (Optional)")
uploaded_files = st.file_uploader(
    "Upload PDF(s) to extend the knowledge base",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    new_docs = []
    for pdf_file in uploaded_files:
        docs = process_pdf(pdf_file)
        new_docs.extend(docs)
    if new_docs:
        # Add new documents to existing vector store
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(VECTOR_DIR)
        st.success("New documents processed and added to knowledge base!")

# Chat interface
st.subheader("Ask a Question")
question = st.text_input("What would you like to know about plumbing codes?")

if question:
    if vectorstore is not None:
        answer = ask_llm(vectorstore, question)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.error("Knowledge base not available. Please contact support.")