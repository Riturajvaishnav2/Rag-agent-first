#!/usr/bin/env python3
"""
Ingest a document into Chroma.
Usage:
  python ingest.py
  python ingest.py --rebuild
"""

import os
import sys
import argparse
from typing import List

from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings

from rag_utils import (
    load_documents,
    get_collection_count,
    get_chroma_client,
    get_langchain_chroma,
    reset_collection,
)

load_dotenv()

# Config
SPEECH_FILE = os.environ.get("SPEECH_FILE", "files/IOTRON INBOUND Traffic.csv")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "langchain")
BATCH_SIZE = max(1, int(os.environ.get("BATCH_SIZE", "500")))


def _add_documents(db, docs: List, batch_size: int):
    total = len(docs)
    for start in range(0, total, batch_size):
        batch = docs[start : start + batch_size]
        texts = [d.page_content for d in batch]
        metadatas = [d.metadata for d in batch]
        db.add_texts(texts, metadatas=metadatas)
        end = min(start + batch_size, total)
        print(f"[i] Ingested {end}/{total} docs")


def ingest(rebuild: bool):
    docs, preview, preview_label = load_documents(SPEECH_FILE, CHUNK_SIZE, CHUNK_OVERLAP)
    if not docs:
        print("[ERROR] No documents loaded. Exiting.", file=sys.stderr)
        sys.exit(2)

    print(f"[i] Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    client, use_http = get_chroma_client(CHROMA_HOST, CHROMA_PORT, PERSIST_DIR)
    if rebuild:
        print(f"[i] Rebuilding collection: {CHROMA_COLLECTION}")
        reset_collection(client, CHROMA_COLLECTION)

    db = get_langchain_chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embeddings=embeddings,
        persist_dir=PERSIST_DIR,
        use_http=use_http,
    )

    existing_count = get_collection_count(db) or 0
    if existing_count > 0 and not rebuild:
        print(
            f"[i] Collection '{CHROMA_COLLECTION}' already has {existing_count} docs. "
            "Use --rebuild to re-ingest."
        )
        return

    print(f"=== {preview_label} ===")
    print(preview.replace("\n", "\\n"))
    print("=== end snippet ===\n")

    print(f"[i] Ingesting {len(docs)} docs into '{CHROMA_COLLECTION}'...")
    _add_documents(db, docs, BATCH_SIZE)

    if not use_http:
        print("[i] Persisting Chroma DB...")
        try:
            db.persist()
        except Exception:
            pass

    print("[i] Ingest complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Delete and re-ingest the collection")
    args = parser.parse_args()
    ingest(args.rebuild)


if __name__ == "__main__":
    main()
