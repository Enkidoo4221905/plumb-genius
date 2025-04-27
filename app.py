import streamlit as st
import os
import pdfplumber
import json
import glob
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
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def debug_message(msg):
    st.sidebar.write(f"[DEBUG] {msg}")

@st.cache_resource
def load_base_knowledge() -> List[Document]:
    """Load the pre-existing knowledge base from JSON and CSV files"""
    documents = []

    # Load JSON files that contain plumbing text
    json_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if "plumbing_text" in os.path.basename(f)
    ]
    debug_message(f"Found JSON files: {[os.path.basename(f) for f in json_files]}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
                debug_message(f"Loaded JSON file: {os.path.basename(json_file)}")
                documents.extend(SPLITTER.create_documents([text]))
        except Exception as e:
            debug_message(f"Error loading JSON {os.path.basename(json_file)}: {e}")

    # Load specific CSV files
    csv_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if any(name in os.path.basename(f) for name in ["ada_data", "plumbing_data"])
    ]
    debug_message(f"Found CSV files: {[os.path.basename(f) for f in csv_files]}")

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                    if text.strip():
                        documents.extend(SPLITTER.create_documents([text]))
                debug_message(f"Loaded CSV file: {os.path.basename(csv_file)}")
        except Exception as e:
            debug_message(f"Error loading CSV {os.path.basename(csv_file)}: {e}")

    debug_message(f"Total documents created: {len(documents)}")
    return documents

def process_pdf(pdf_file) -> List[Document]:
    """Process new PDF uploads"""
    docs = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    docs.extend(SPLITTER.create_documents([text], metadatas=[{"source": f"{pdf_file.name}_page_{i+1}"}]))
        debug_message(f"Processed PDF: {pdf_file.name}, pages: {len(pdf.pages)}, docs created: {len(docs)}")
    except Exception as e:
        debug_message(f"Error processing PDF {pdf_file.name}: {e}")
    return docs

@st.cache_resource
def initialize_vectorstore():
    """Initialize or load the vector store with base knowledge"""
    debug_message("Initializing vector store...")

    if os.path.exists(VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
            debug_message("Successfully loaded existing vector store")
            return vectorstore
        except Exception as e:
            debug_message(f"Failed to load vector store: {e}")

    # If vector store doesn't exist, create it from base knowledge
    debug_message("Creating new vector store from base knowledge...")
    base_docs = load_base_knowledge()
    if base_docs:
        vectorstore = FAISS.from_documents(base_docs, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        debug_message("Successfully created and saved new vector store")
        return vectorstore

    debug_message("No documents available to create vector store")
    return None

def ask_llm(vectorstore, question: str, k: int = 5) -> str:
    try:
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"You are a plumbing expert with knowledge of the 2021 IPC.\nUse the following context to answer accurately:\n{context}\nQuestion: {question}\nAnswer:"
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        debug_message(f"LLM query failed: {e}")
        return "Sorry, I couldn't process your question."

# --- Streamlit UI ---

st.title("Plumb Genius: Plumbing Code Chatbot")
st.sidebar.title("Debug Information")

# Initialize vector store with base knowledge
vectorstore = initialize_vectorstore()

if vectorstore is not None:
    st.sidebar.success("Knowledge base is loaded and ready!")
else:
    st.sidebar.error("Knowledge base could not be loaded")

# Chat interface
st.subheader("Ask a Question")
question = st.text_input("What would you like to know about plumbing codes?")

if question:
    if vectorstore is not None:
        answer = ask_llm(vectorstore, question)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.error("Knowledge base not available. Please contact support.")

# PDF upload section
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
        if vectorstore is None:
            debug_message("Creating new vector store from uploaded PDFs...")
            vectorstore = FAISS.from_documents(new_docs, embeddings)
        else:
            debug_message("Adding new documents to existing vector store...")
            vectorstore.add_documents(new_docs)
        vectorstore.save_local(VECTOR_DIR)
        st.success("New documents processed and added to knowledge base!")
        debug_message(f"Knowledge base now contains new documents. Please refresh to use the updated knowledge base.")