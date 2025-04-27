import streamlit as st
import os
import pdfplumber
import json
import glob
import csv
import time
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants with optimized settings
VECTOR_DIR = "./vector_store"
DATA_DIR = "./data"
BATCH_SIZE = 50  # Smaller batch size to avoid rate limits
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Larger chunks to reduce total number of embeddings
    chunk_overlap=100,
    length_function=len,
)

# Initialize OpenAI models with retry settings
embeddings = OpenAIEmbeddings(
    chunk_size=BATCH_SIZE,  # Process in smaller batches
    max_retries=10,  # More retries for resilience
    request_timeout=60,  # Longer timeout
)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_retries=10,
    request_timeout=60,
)

def debug_message(msg):
    st.sidebar.write(f"[DEBUG] {msg}")

def smart_truncate(text: str, max_length: int = 8000) -> str:
    """Intelligently truncate text to avoid embedding very long documents."""
    if len(text) <= max_length:
        return text
    return text[:max_length]

@st.cache_resource
def load_base_knowledge(max_files: int = None) -> List[Document]:
    """Load documents with smart batching and filtering"""
    documents = []

    # Load JSON files that contain plumbing text
    json_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.json"))
        if "plumbing_text" in os.path.basename(f)
    ]
    debug_message(f"Found JSON files: {[os.path.basename(f) for f in json_files]}")

    # Process limited number of files if specified
    if max_files:
        json_files = json_files[:max_files]

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
                # Smart truncate long texts
                text = smart_truncate(text)
                debug_message(f"Processing JSON file: {os.path.basename(json_file)}")
                chunks = SPLITTER.create_documents([text])
                documents.extend(chunks)
                debug_message(f"Added {len(chunks)} chunks from {os.path.basename(json_file)}")
        except Exception as e:
            debug_message(f"Error loading JSON {os.path.basename(json_file)}: {e}")

    # Load specific CSV files
    csv_files = [
        f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if any(name in os.path.basename(f) for name in ["ada_data", "plumbing_data"])
    ]
    if max_files:
        csv_files = csv_files[:max_files]

    debug_message(f"Found CSV files: {[os.path.basename(f) for f in csv_files]}")

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                    if text.strip():
                        rows.append(text)

                # Process rows in batches
                for i in range(0, len(rows), 10):  # Process 10 rows at a time
                    batch = rows[i:i+10]
                    combined_text = "\n\n".join(batch)
                    chunks = SPLITTER.create_documents([combined_text])
                    documents.extend(chunks)
                debug_message(f"Processed CSV file: {os.path.basename(csv_file)}")
        except Exception as e:
            debug_message(f"Error loading CSV {os.path.basename(csv_file)}: {e}")

    debug_message(f"Total documents created: {len(documents)}")
    return documents

def process_pdf(pdf_file) -> List[Document]:
    """Process new PDF uploads with smart chunking"""
    docs = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text_chunks = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
                    if len(text_chunks) >= 5:  # Combine 5 pages at a time
                        combined_text = "\n\n".join(text_chunks)
                        docs.extend(SPLITTER.create_documents([combined_text]))
                        text_chunks = []

            # Process any remaining chunks
            if text_chunks:
                combined_text = "\n\n".join(text_chunks)
                docs.extend(SPLITTER.create_documents([combined_text]))

        debug_message(f"Processed PDF: {pdf_file.name}, created {len(docs)} chunks")
    except Exception as e:
        debug_message(f"Error processing PDF {pdf_file.name}: {e}")
    return docs

def batch_create_vectorstore(documents: List[Document], batch_size: int = BATCH_SIZE):
    """Create vector store in batches to handle rate limits"""
    vectorstore = None
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        debug_message(f"Processing batch {(i//batch_size)+1}/{total_batches}")

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                temp_vectorstore = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(temp_vectorstore)

            # Add a small delay between batches
            time.sleep(1)
        except Exception as e:
            debug_message(f"Error in batch {(i//batch_size)+1}: {e}")
            # Continue with next batch instead of failing completely
            continue

    return vectorstore

@st.cache_resource
def initialize_vectorstore():
    """Initialize or load the vector store with batched processing"""
    debug_message("Initializing vector store...")

    if os.path.exists(VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
            debug_message("Successfully loaded existing vector store")
            return vectorstore
        except Exception as e:
            debug_message(f"Failed to load vector store: {e}")

    debug_message("Creating new vector store from base knowledge...")
    base_docs = load_base_knowledge(max_files=2)  # Start with limited files
    if base_docs:
        try:
            vectorstore = batch_create_vectorstore(base_docs)
            if vectorstore:
                vectorstore.save_local(VECTOR_DIR)
                debug_message("Successfully created and saved new vector store")
                return vectorstore
        except Exception as e:
            debug_message(f"Error during vector store creation: {e}")
            st.error("Error creating vector store. Using reduced functionality.")
            return None

    debug_message("No documents available to create vector store")
    return None

def ask_llm(vectorstore, question: str, k: int = 3) -> str:
    """Generate answer with optimized context retrieval"""
    try:
        docs = vectorstore.similarity_search(question, k=k)
        # Combine and truncate context to stay within token limits
        context = "\n".join([doc.page_content for doc in docs])
        context = smart_truncate(context, 4000)  # Limit context size

        prompt = f"""You are a plumbing expert with knowledge of the 2021 IPC.
Use the following context to answer accurately and concisely:
{context}
Question: {question}
Answer:"""

        return llm.invoke(prompt).content.strip()
    except Exception as e:
        debug_message(f"LLM query failed: {e}")
        return "Sorry, I couldn't process your question. Please try again."

# Streamlit UI
st.title("Plumb Genius: Plumbing Code Chatbot")
st.sidebar.title("Debug Information")

# Initialize vector store with base knowledge
vectorstore = initialize_vectorstore()

if vectorstore is not None:
    st.sidebar.success("Knowledge base is loaded and ready!")
else:
    st.sidebar.warning("Limited knowledge base available")

# Chat interface
st.subheader("Ask a Question")
question = st.text_input("What would you like to know about plumbing codes?")

if question:
    if vectorstore is not None:
        with st.spinner("Generating answer..."):
            answer = ask_llm(vectorstore, question)
            st.markdown(f"**Answer:** {answer}")
    else:
        st.error("Knowledge base is limited. Please try a simpler question.")

# PDF upload section
st.subheader("Upload Additional Documents (Optional)")
uploaded_files = st.file_uploader(
    "Upload PDF(s) to extend the knowledge base",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        new_docs = []
        for pdf_file in uploaded_files:
            docs = process_pdf(pdf_file)
            new_docs.extend(docs)

        if new_docs:
            try:
                if vectorstore is None:
                    vectorstore = batch_create_vectorstore(new_docs)
                else:
                    # Process new documents in batches
                    temp_vectorstore = batch_create_vectorstore(new_docs)
                    if temp_vectorstore:
                        vectorstore.merge_from(temp_vectorstore)

                vectorstore.save_local(VECTOR_DIR)
                st.success("Documents processed and added to knowledge base!")
                debug_message("Knowledge base updated with new documents")
            except Exception as e:
                st.error("Error adding documents to knowledge base")
                debug_message(f"Error: {e}")