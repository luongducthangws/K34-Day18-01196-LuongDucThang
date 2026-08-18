"""Shared configuration for Lab 18."""

import os, sys
from dotenv import load_dotenv

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Fix HF_HOME if pointing to a non-existent drive (e.g. unmounted Google Drive G:)
hf_home = os.environ.get("HF_HOME", "")
if hf_home:
    drive = os.path.splitdrive(hf_home)[0]
    if drive and not os.path.exists(drive + "\\"):
        os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

# --- API Keys & LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("OPENROUTER_BASE_URL", None))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
