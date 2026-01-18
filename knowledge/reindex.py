from .embeddings import compute_embedding, save_embedding, build_faiss_index, ensure_index

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Se reconstruiește indexul FAISS din baza de date...")
    build_faiss_index() # Colectează toți vectorii și îi salvează pe disc
    ensure_index()      # Verifică integritatea indexului
    logging.info("Reindexare finalizată cu succes!")