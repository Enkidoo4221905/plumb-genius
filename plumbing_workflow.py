from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, END
import pandas as pd
import json
import os
import glob
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.docstore.document import Document
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = os.path.expanduser("~/plumbgenius/extracted_texts")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
VECTOR_DIR = "./vector_store"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL)

class TableEntry(BaseModel):
    source_pdf: str
    table_number: int
    location_2021: Optional[str]
    size: Optional[str]
    flow: Optional[str]
    pressure: Optional[str]
    data: dict

    @classmethod
    def from_row(cls, row):
        row_dict = row.to_dict()
        def clean_value(val):
            return None if pd.isna(val) else str(val)
        return cls(
            source_pdf=row_dict.get("Source PDF", ""),
            table_number=int(row_dict.get("Table Number", 0)),
            location_2021=clean_value(row_dict.get("2021 LOCATION")),
            size=clean_value(row_dict.get("0")),
            flow=clean_value(row_dict.get("1")),
            pressure=clean_value(row_dict.get("2")),
            data={k: clean_value(v) for k, v in row_dict.items() if k not in ["Source PDF", "Table Number", "2021 LOCATION", "0", "1", "2"]}
        )

class TextEntry(BaseModel):
    page_number: int
    text: str
    tables: List[list]

class PlumbingData(BaseModel):
    tables: List[TableEntry]
    text: List[TextEntry]

# Load data
def load_plumbing_data():
    logger.info("Loading plumbing data...")
    csv_files = glob.glob(os.path.join(DATA_PATH, "*_data*.csv"))
    if not csv_files:
        logger.warning("No CSV files found.")
        df = pd.DataFrame()
    else:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    table_entries = [TableEntry.from_row(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating table rows") if not df.empty]
    json_path = os.path.join(DATA_PATH, "plumbing_text.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            text_data_raw = json.load(f)
        text_entries = [TextEntry(**entry) for entry in tqdm(text_data_raw, desc="Validating text entries")]
    else:
        logger.warning("plumbing_text.json not found.")
        text_entries = []
    return PlumbingData(tables=table_entries, text=text_entries)

# Vector store
def load_or_build_vector_store(data: PlumbingData):
    if os.path.exists(VECTOR_DIR):
        try:
            vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing vector store.")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
    docs = [Document(page_content=f"Table {e.table_number}: Size={e.size}, Flow={e.flow}, Pressure={e.pressure}", 
                    metadata={"source": e.source_pdf}) for e in data.tables]
    docs += [Document(page_content=e.text, metadata={"page": e.page_number}) for e in data.text]
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    logger.info("Built and saved new vector store.")
    return vectorstore

class PlumbingState(TypedDict):
    data: PlumbingData
    query: str
    response: Optional[str]

def start_node(state: PlumbingState) -> PlumbingState:
    logger.info(f"Starting with query: {state['query']}")
    return state

def ollama_node(state: PlumbingState) -> PlumbingState:
    vectorstore = load_or_build_vector_store(state["data"])
    docs = vectorstore.similarity_search(state["query"], k=5)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Given this 2021 IPC data:\n{context}\n\nAnswer: {state['query']}"
    state["response"] = llm.invoke(prompt).content.strip()
    return state

workflow = StateGraph(PlumbingState)
workflow.add_node("start", start_node)
workflow.add_node("ollama", ollama_node)
workflow.add_edge("start", "ollama")
workflow.add_edge("ollama", END)
workflow.set_entry_point("start")
app = workflow.compile()

def run_workflow(query: str) -> str:
    plumbing_data = load_plumbing_data()
    initial_state = {"data": plumbing_data, "query": query}
    result = app.invoke(initial_state)
    return result["response"]

if __name__ == "__main__":
    plumbing_data = load_plumbing_data()
    if not plumbing_data.tables and not plumbing_data.text:
        logger.error("No data loaded. Please provide CSV and/or JSON files.")
    else:
        result = run_workflow("What are the key pipe sizing requirements in the 2021 IPC?")
        logger.info(f"Final response: {result}")