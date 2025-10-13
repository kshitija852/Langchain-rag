# -----------------------------
# create_database.py
# -----------------------------
import os
import time
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # new import (no deprecation)

# -----------------------------
#  Load environment variables
# -----------------------------
load_dotenv()  # loads from .env in project root

# -----------------------------
# Data setup
# -----------------------------
data_path = "Data"
chroma_path = "chroma"

# -----------------------------
# Safe delete function for Chroma DB
# -----------------------------
def safe_delete_chroma(path, retries=5, delay=2):
    """Safely delete a Chroma DB folder, retrying if Windows locks files."""
    if not os.path.exists(path):
        return
    for i in range(retries):
        try:
            shutil.rmtree(path)
            print(f"Deleted existing Chroma DB at {path}")
            return
        except PermissionError:
            print(f"Attempt {i+1}: Chroma still in use, retrying...")
            time.sleep(delay)
    print(f" Could not delete {path} — please close any running Python/Chroma processes.")

# -----------------------------
#  Delete existing database before creating new one
# -----------------------------
safe_delete_chroma(chroma_path)

# -----------------------------
#  Load and split Markdown documents
# -----------------------------
def load_data():
    loader = DirectoryLoader(data_path, glob="*.md")
    docs = loader.load()
    return docs

docs = load_data()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(docs)
print(f"📄 Split {len(docs)} documents into {len(chunks)} chunks.")

# -----------------------------
#  Initialize embeddings (free + local)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
#  Create Chroma DB from chunks
# -----------------------------
db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
# db.persist()
# The data got successfully added onto the chroma.
print(f"Persisted database successfully to: {chroma_path}")

