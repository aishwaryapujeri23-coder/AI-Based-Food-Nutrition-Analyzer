"""
rag_chain.py
------------
Loads the FAISS vector store built by ingest.py and wires it up to a
model served by Groq's cloud API to answer nutrition questions
using Retrieval-Augmented Generation (RAG).

Why Groq instead of local Ollama:
    Ollama requires downloading a multi-GB model and running a local
    server, which needs real disk space and RAM. Groq runs open-weight
    language models in the cloud, with no local
    storage needed at all. The retrieval half of this pipeline (FAISS +
    Sentence Transformers) still runs 100% locally either way — only the
    final answer-generation step happens on Groq's servers.

Requires:
    - You have already run `python ingest.py` at least once.
    - A free Groq API key from https://console.groq.com/keys
      (paste it into your .env file as GROQ_API_KEY=...)
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PROMPT_TEMPLATE = """You are a helpful, careful nutrition assistant.
Use ONLY the context below to answer the question. If the context does not
contain the answer, say "I don't have enough information in my knowledge
base to answer that" instead of guessing.

Context:
{context}

Question: {question}

Answer in a clear, friendly, and concise way:"""


def load_rag_chain():
    """
    Loads the vector store + Groq-hosted LLM and returns a ready-to-use
    RetrievalQA chain.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"No vector store found at '{VECTORSTORE_DIR}/'. "
            "Run `python ingest.py` first to build it."
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this file ourselves
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def ask_question(qa_chain, question: str) -> dict:
    """
    Runs a question through the RAG chain.
    Returns {"answer": str, "sources": [str, ...]}
    """
    result = qa_chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        page = doc.metadata.get("page", "?")
        source_file = os.path.basename(doc.metadata.get("source", "unknown.pdf"))
        sources.append(f"{source_file} (page {page})")

    return {
        "answer": result["result"],
        "sources": list(dict.fromkeys(sources)),  # de-duplicate, keep order
    }


if __name__ == "__main__":
    chain = load_rag_chain()
    while True:
        q = input("\nAsk a nutrition question (or 'quit'): ")
        if q.lower() in ("quit", "exit"):
            break
        response = ask_question(chain, q)
        print("\nAnswer:", response["answer"])
        print("Sources:", response["sources"])