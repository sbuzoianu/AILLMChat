import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "knowledge.db")

SBERT_MODEL = os.environ.get("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
# FLAN_MODEL = os.environ.get("FLAN_MODEL", "google/flan-t5-small")  # small e mai rapid pe CPU
# FLAN_MODEL = "MBZUAI/LaMini-T5-738M" # optimizat pentru limba romana
FLAN_MODEL = "google/flan-t5-base"

# Configuri suplimentare
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "knowledge", "faiss_files")
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
