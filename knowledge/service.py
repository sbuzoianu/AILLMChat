# knowledge/service.py
from database.db import get_connection
from .embeddings import compute_embedding, save_embedding, build_faiss_index, ensure_index

def add_knowledge(subject, grade, content):
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO knowledge (subject, grade, content) VALUES (?, ?, ?)",
        (subject, grade, content)
    )
    knowledge_id = cur.lastrowid
    con.commit()
    con.close()

    emb = compute_embedding(content)
    save_embedding(knowledge_id, emb)
    # Pentru set mic: rebuild index. Pentru dataset mare: append logic.
    build_faiss_index()
    ensure_index()

def get_knowledge(subject, grade):
    con = get_connection()
    cur = con.cursor()
    cur.execute(
        "SELECT content FROM knowledge WHERE subject=? AND grade=?",
        (subject, grade)
    )
    result = cur.fetchone()
    con.close()
    return result[0] if result else None
