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

# Disable Chroma telemetry to avoid posthog capture errors in this environment.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from dotenv import load_dotenv
import pandas as pd
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

# Config
SPEECH_FILE = os.environ.get("SPEECH_FILE", "files/IOTRON INBOUND Traffic.csv")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "/data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K = int(os.environ.get("TOP_K", "3"))
OLLAMA_URL = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "langchain")
EXCEL_EXTENSIONS = (".xlsx", ".xls", ".xlsm", ".xltx", ".xltm")
META_KEYS = {
    "Traffic Period",
    "Continent",
    "Country",
    "Operator Name",
    "PMN",
    "Service Type",
    "Service Unit",
    "Call Destination",
}

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

def build_docs_from_dataframe(df: pd.DataFrame, extra_metadata: dict | None = None) -> List[Document]:
    docs: List[Document] = []
    base_metadata = extra_metadata or {}
    for idx, row in df.iterrows():
        parts: List[str] = []
        metadata = dict(base_metadata)
        try:
            metadata["row"] = int(idx)
        except Exception:
            metadata["row"] = str(idx)
        if "sheet" in metadata:
            parts.append(f"Sheet: {metadata['sheet']}")
        for col, val in row.items():
            if pd.isna(val):
                continue
            text_val = str(val).strip()
            if not text_val or text_val.lower() == "nan":
                continue
            parts.append(f"{col}: {text_val}")
            if col in META_KEYS:
                metadata[col] = text_val
        if parts:
            docs.append(Document(page_content="; ".join(parts), metadata=metadata))
    return docs

def load_csv_documents(path: str) -> tuple[List[Document], str]:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    df = None
    used_encoding = None
    last_error: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break

    if df is None:
        print(f"[ERROR] Could not read CSV file: {last_error}", file=sys.stderr)
        return [], ""

    if used_encoding and used_encoding != "utf-8":
        print(f"[i] CSV decoded with fallback encoding: {used_encoding}")

    preview = df.head(5).to_csv(index=False).strip()
    return build_docs_from_dataframe(df), preview

def load_excel_documents(path: str) -> tuple[List[Document], str, str]:
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception as e:
        print(f"[ERROR] Could not read Excel file: {e}", file=sys.stderr)
        return [], "", ""

    docs: List[Document] = []
    preview = ""
    preview_sheet = ""
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        if not preview:
            preview = df.head(5).to_csv(index=False).strip()
            preview_sheet = sheet_name
        docs.extend(build_docs_from_dataframe(df, {"sheet": sheet_name}))

    return docs, preview, preview_sheet

def load_documents(path: str) -> tuple[List[Document], str, str]:
    if not os.path.exists(path):
        print(f"[ERROR] speech file not found at: {path}", file=sys.stderr)
        return [], "", ""

    lower_path = path.lower()
    if lower_path.endswith(".csv"):
        docs, preview = load_csv_documents(path)
        return docs, preview, "CSV preview (first 5 rows)"

    if lower_path.endswith(EXCEL_EXTENSIONS):
        docs, preview, preview_sheet = load_excel_documents(path)
        label = "Excel preview (first 5 rows)"
        if preview_sheet:
            label = f"{label} | sheet: {preview_sheet}"
        return docs, preview, label

    text = read_speech(path)
    if not text:
        return [], "", ""
    return chunk_text(text), text[:1000].replace("\n", "\\n"), "file snippet (first 1000 chars)"

def _get_collection_count(db: Chroma) -> int | None:
    try:
        return db._collection.count()
    except Exception:
        try:
            return len(db._collection.get()["ids"])
        except Exception:
            return None


def build_or_load_chroma(docs: List[Document], embeddings) -> Chroma:
    use_http = bool(CHROMA_HOST)
    db = None

    if use_http:
        print(f"[i] Using Chroma HTTP server at {CHROMA_HOST}:{CHROMA_PORT} (collection: {CHROMA_COLLECTION})")
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            db = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=None,
                client=client,
            )
            db._embedding_function = embeddings
        except Exception as e:
            print(
                f"[ERROR] Could not connect to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}: {e}",
                file=sys.stderr,
            )
            sys.exit(3)
    else:
        # ensure persist dir exists
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # Always open the persisted collection; add chunks only if empty.
        try:
            db = Chroma(
                collection_name=CHROMA_COLLECTION,
                persist_directory=PERSIST_DIR,
                embedding_function=None,
            )
            db._embedding_function = embeddings
            existing_count = _get_collection_count(db)
            if existing_count and existing_count > 0:
                print(f"[i] Loading existing Chroma DB from {PERSIST_DIR} (docs: {existing_count})")
                return db
        except Exception as e:
            print(f"[!] Could not load existing Chroma DB: {e}")
            db = None

    if db is not None:
        existing_count = _get_collection_count(db)
        if existing_count and existing_count > 0:
            if use_http:
                print(f"[i] Loading existing Chroma collection '{CHROMA_COLLECTION}' (docs: {existing_count})")
            else:
                print(f"[i] Loading existing Chroma DB from {PERSIST_DIR} (docs: {existing_count})")
            return db

    print("[i] Chroma DB is empty; adding all chunks once at startup...")
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    if db is None:
        print("[ERROR] Chroma DB not initialized.", file=sys.stderr)
        sys.exit(3)
    db.add_texts(texts, metadatas=metadatas)
    if not use_http:
        print("[i] Persisting Chroma DB...")
        try:
            db.persist()
        except Exception:
            pass
        print(f"[i] Created and persisted Chroma DB at: {PERSIST_DIR}")
    else:
        print(f"[i] Created Chroma collection '{CHROMA_COLLECTION}' on {CHROMA_HOST}:{CHROMA_PORT}")
    return db

def debug_and_run(query: str, debug: bool):
    print(f"[i] Using embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    docs, preview, preview_label = load_documents(SPEECH_FILE)
    if not docs:
        print("[ERROR] No documents loaded. Exiting.", file=sys.stderr)
        sys.exit(2)

    print(f"=== {preview_label} ===")
    print(preview.replace("\n", "\\n"))
    print("=== end snippet ===\n")

    lower_path = SPEECH_FILE.lower()
    if lower_path.endswith(".csv") or lower_path.endswith(EXCEL_EXTENSIONS):
        print(f"[i] Created {len(docs)} row docs from table file.")
    else:
        print(f"[i] Created {len(docs)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    if len(docs) > 0:
        print("=== first document (full) ===")
        print(docs[0].page_content)
        print("=== end first document ===\n")

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
