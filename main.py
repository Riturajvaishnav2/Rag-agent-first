#!/usr/bin/env python3
"""
Debug main.py
- Prints file read status
- Shows chunking results
- Shows Chroma index document count
- Shows top-k retrieved chunks (full text) before calling the LLM
- Calls Ollama for final answer (no fallback)
Usage:
  python main.py -q "Summarize the speech"
  python main.py --debug -q "Summarize the speech"
"""

import os
import sys
import argparse
import time
from typing import List

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

# Config
SPEECH_FILE = os.environ.get("SPEECH_FILE", "files/AI-for-Education-RAG.pdf")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K = int(os.environ.get("TOP_K", "3"))
OLLAMA_URL = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")

def read_speech(path: str) -> str:
    if not os.path.exists(path):
        print(f"[ERROR] speech file not found at: {path}", file=sys.stderr)
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    return data

def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    return docs

def build_or_load_chroma(docs: List[Document], embeddings) -> Chroma:
    # ensure persist dir exists
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # If directory empty -> create new, else load
    try:
        if any(os.scandir(PERSIST_DIR)):
            print(f"[i] Loading existing Chroma DB from {PERSIST_DIR}")
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            # Try to print collection count if available
            try:
                ccount = len(db._collection.get()["metadatas"])
                print(f"[i] Chroma collection loaded; approx count (via metadata): {ccount}")
            except Exception:
                pass
            return db
    except Exception as e:
        print(f"[!] Could not load existing Chroma DB: {e}")

    print("[i] Creating new Chroma DB from docs...")
    texts = [d.page_content for d in docs]
    db = Chroma.from_texts(texts, embedding=embeddings, persist_directory=PERSIST_DIR)
    
    # Explicitly persist after documents are added
    print(f"[i] Persisting Chroma DB...")
    db.persist()

    try:
        db.persist()
    except Exception:
        pass

    print(f"[i] Created and persisted Chroma DB at: {PERSIST_DIR}")
    return db

def debug_and_run(query: str, debug: bool):
    print(f"[i] Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Read speech
    text = read_speech(SPEECH_FILE)
    if not text:
        print("[ERROR] No text loaded. Exiting.", file=sys.stderr)
        sys.exit(2)

    # Print initial snippet
    print("=== files/AI-for-Education-RAG.pdf (first 1000 chars) ===")
    print(text[:1000].replace("\n", "\\n"))
    print("=== end snippet ===\n")

    docs = chunk_text(text)
    print(f"[i] Created {len(docs)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    if len(docs) > 0:
        print("=== first chunk (full) ===")
        print(docs[0].page_content)
        print("=== end first chunk ===\n")

    # Build/load Chroma
    db = build_or_load_chroma(docs, embeddings)

    # Show collection size via retriever: try similarity search with empty string or a word to get count
    try:
        # There isn't a direct 'count' API in all chroma versions; try an approximate
        # We'll index the texts length as sanity check
        print("[i] Attempting to inspect stored documents via a search for a common token 'the' ...")
        results = db.similarity_search("the", k=1)
        print(f"[i] similarity_search('the', k=1) returned {len(results)} docs")
    except Exception as e:
        print(f"[!] Could not perform inspection search: {e}")

    # Prepare retriever and show top-k retrieved chunks for the query BEFORE calling LLM
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    retrieved = retriever.get_relevant_documents(query)
    print(f"[i] Retrieved top {len(retrieved)} docs for query: '{query}'")
    for i, d in enumerate(retrieved, 1):
        print(f"\n--- Retrieved doc {i} (len {len(d.page_content)} chars) ---")
        print(d.page_content)
        print(f"--- end doc {i} ---\n")

    # Now call Ollama
    print(f"[i] Initializing Ollama at {OLLAMA_URL}")
    try:
        llm = Ollama(model="mistral", base_url=OLLAMA_URL)
    except Exception as e:
        print(f"[ERROR] Could not initialize Ollama LLM: {e}", file=sys.stderr)
        sys.exit(3)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("[i] Running query through LLM...")
    start = time.time()
    out = qa.run(query)
    dur = time.time() - start
    print(f"[i] LLM finished in {dur:.2f}s\n\n=== LLM ANSWER ===\n{out}\n=== END ANSWER ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()
    debug_and_run(args.query, args.debug)

if __name__ == "__main__":
    main()
