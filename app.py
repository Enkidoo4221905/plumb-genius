import streamlit as st
import os
import pdfplumber
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
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize OpenAI models
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

@st.cache_resource
def load_vectorstore():
    if os.path.exists(VECTOR_DIR):
        try:
            return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Failed to load vector store: {e}")
            return None
    return None

def process_pdf(pdf_file) -> List[Document]:
    """Extract text from a PDF and split into LangChain Documents."""
    docs = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.extend(SPLITTER.create_documents([text], metadatas=[{"source": f"{pdf_file.name}_page_{i+1}"}]))
    return docs

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

# Upload PDF(s)
uploaded_files = st.file_uploader(
    "Upload PDF(s) to add to the knowledge base",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for pdf_file in uploaded_files:
        docs = process_pdf(pdf_file)
        all_docs.extend(docs)
    if all_docs:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        st.success("PDFs processed and vector store updated!")
    else:
        st.warning("No text found in uploaded PDFs.")

# Always try to load the vector store
vectorstore = load_vectorstore()

if vectorstore is not None:
    question = st.text_input("Ask a plumbing question:")
    if question:
        answer = ask_llm(vectorstore, question)
        st.markdown(f"**Answer:** {answer}")
else:
    st.info("Please upload PDFs to build the knowledge base.")