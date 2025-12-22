#!/usr/bin/env python3
"""
Query Chroma and send the retrieved context to Ollama.
Usage:
  python query.py -q "Your question"
  python query.py --debug -q "Your question"
"""

import os
import sys
import argparse
import time

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

from rag_utils import (
    get_collection_count,
    get_chroma_client,
    get_langchain_chroma,
)

load_dotenv()

# Config
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))
OLLAMA_URL = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "langchain")


def run_query(query: str, debug: bool):
    print(f"[i] Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    client, use_http = get_chroma_client(CHROMA_HOST, CHROMA_PORT, PERSIST_DIR)
    db = get_langchain_chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embeddings=embeddings,
        persist_dir=PERSIST_DIR,
        use_http=use_http,
    )

    existing_count = get_collection_count(db) or 0
    if existing_count == 0:
        print(
            f"[ERROR] Collection '{CHROMA_COLLECTION}' is empty. "
            "Run ingest.py first.",
            file=sys.stderr,
        )
        sys.exit(2)

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    if debug:
        retrieved = retriever.get_relevant_documents(query)
        print(f"[i] Retrieved top {len(retrieved)} docs for query: '{query}'")
        for i, d in enumerate(retrieved, 1):
            print(f"\n--- Retrieved doc {i} (len {len(d.page_content)} chars) ---")
            print(d.page_content)
            print(f"--- end doc {i} ---\n")

    print(f"[i] Initializing Ollama at {OLLAMA_URL} (model: {OLLAMA_MODEL})")
    try:
        llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
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
    run_query(args.query, args.debug)


if __name__ == "__main__":
    main()
