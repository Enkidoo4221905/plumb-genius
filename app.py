import streamlit as st
import os
import pdfplumber
import json
import glob
import csv
import hashlib
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Changed from Ollama
from langchain.docstore.document import Document
import time
from dotenv import load_dotenv  # Added for .env support

# Load environment variables
load_dotenv()

# Constants
VECTOR_DIR = "./vector_store"
DATA_PATH = "data"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize OpenAI models (replaced Ollama)
embeddings = OpenAIEmbeddings()  # Uses 'text-embedding-ada-002' by default
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

processed_pdfs = set()

# Rest of your functions remain the same
@st.cache_resource
def load_vectorstore():
    if os.path.exists(VECTOR_DIR):
        try:
            return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            return None
    return None

# Your other functions (load_all_data, process_pdf) remain exactly the same

def ask_llm(vectorstore, question: str, k: int = 5) -> str:
    try:
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"You are a plumbing expert with knowledge of the 2021 IPC.\nUse the following context to answer accurately:\n{context}\nQuestion: {question}\nAnswer:"
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        st.error(f"LLM query failed: {e}")
        return "Sorry, I couldn't process your question."

# UI section remains exactly the same as your original code