#!/usr/bin/env python3
"""
Shared helpers for loading data and connecting to Chroma.
"""

import os
import sys
from typing import List

import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

# Disable Chroma telemetry to avoid posthog capture errors in this environment.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

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


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


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


def load_documents(path: str, chunk_size: int, chunk_overlap: int) -> tuple[List[Document], str, str]:
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
    snippet = text[:1000].replace("\n", "\\n")
    return chunk_text(text, chunk_size, chunk_overlap), snippet, "file snippet (first 1000 chars)"


def get_collection_count(db: Chroma) -> int | None:
    try:
        return db._collection.count()
    except Exception:
        try:
            return len(db._collection.get()["ids"])
        except Exception:
            return None


def get_chroma_client(chroma_host: str, chroma_port: int, persist_dir: str):
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=chroma_port), True
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir), False


def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def get_langchain_chroma(
    client,
    collection_name: str,
    embeddings,
    persist_dir: str,
    use_http: bool,
) -> Chroma:
    if use_http:
        db = Chroma(
            collection_name=collection_name,
            client=client,
            embedding_function=None,
        )
    else:
        db = Chroma(
            collection_name=collection_name,
            client=client,
            persist_directory=persist_dir,
            embedding_function=None,
        )
    db._embedding_function = embeddings
    return db
