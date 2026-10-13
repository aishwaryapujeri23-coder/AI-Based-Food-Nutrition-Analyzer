"""
ingest.py
---------
Run this ONCE (and again any time you add/change files) to build the FAISS
vector store from the knowledge base in the `data/` folder.

Usage:
    python ingest.py

What it does:
1. Loads every PDF and TXT file in data/
2. Splits them into overlapping chunks
3. Embeds each chunk using a local Sentence Transformer model
   (all-MiniLM-L6-v2 — small, fast, free, runs on CPU)
4. Stores the embeddings in a FAISS index on disk (vectorstore/)
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: str):
    """Load every PDF and TXT file found in data_dir into LangChain Document objects."""
    documents = []
    all_files = os.listdir(data_dir)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
    txt_files = [f for f in all_files if f.lower().endswith(".txt")]

    if not pdf_files and not txt_files:
        print(f"No PDF or TXT files found in '{data_dir}/'.")
        print("Add at least one nutrition-related .pdf or .txt file there and re-run this script.")
        sys.exit(1)

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"Loading PDF: {filename}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    for filename in txt_files:
        path = os.path.join(data_dir, filename)
        print(f"Loading TXT: {filename}")
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} document(s)/page(s) from "
          f"{len(pdf_files)} PDF(s) and {len(txt_files)} TXT file(s).")
    return documents


def split_documents(documents):
    """Split loaded documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and save a FAISS index to disk."""
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building FAISS index (this may take a minute)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to '{VECTORSTORE_DIR}/'.")


def main():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    build_vectorstore(chunks)
    print("\nDone! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()