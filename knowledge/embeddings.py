# knowledge/embeddings.py
# căutarea semantică — sistemul înțelege sensul întrebării, chiar dacă utilizatorul folosește cuvinte diferite de cele din baza de date.
import os
import numpy as np
import faiss # Facebook AI Similarity Search
from sentence_transformers import SentenceTransformer
from database.db import get_connection

MODEL_NAME = os.environ.get("SBERT_MODEL", "all-mpnet-base-v2")  # bun multilingual/eng; pentru română poți folosi 'paraphrase-multilingual-MiniLM-L12-v2' sau model dedicat
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss.index")
FAISS_MAP_PATH = os.path.join(os.path.dirname(__file__), "faiss_map.npy")

_model = None
_index = None
_id_map = None
_dim = None

def _get_model():
    global _model
    if _model is None:
        # Ia o frază (ex: „Cum funcționează gravitația?”) și o transformă într-un vector
        # Fraze cu sens similar (ex: „Forța de atracție a Pământului”) vor genera liste de numere foarte apropiate matematic într-un spațiu multidimensional.
        _model = SentenceTransformer(MODEL_NAME) 
    return _model

def compute_embedding(text: str) -> np.ndarray:
    model = _get_model()
    emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
    emb = emb.astype('float32')
    return emb

def _to_blob(vec: np.ndarray):
    return vec.tobytes()

def _from_blob(blob, dim):
    arr = np.frombuffer(blob, dtype='float32')
    return arr.reshape((dim,))
# Salvează vectorul în baza de date sub formă de BLOB (date binare brute)
def save_embedding(knowledge_id: int, vector: np.ndarray):
    con = get_connection()
    cur = con.cursor()
    cur.execute("REPLACE INTO embeddings (id, vector, dim) VALUES (?, ?, ?)",
                (knowledge_id, vector.tobytes(), int(vector.shape[0])))
    con.commit()
    con.close()

# Colectează toți vectorii salvați în baza de date, îi „împachetează” într-o structură specială numită Index și o salvează pe disc
def build_faiss_index():
    global _index, _id_map, _dim
    con = get_connection()
    cur = con.cursor()
    cur.execute("SELECT id, vector, dim FROM embeddings")
    rows = cur.fetchall()
    con.close()

    if not rows:
        _index = None
        _id_map = np.array([], dtype=int)
        return

    ids = []
    vectors = []
    for r in rows:
        kid, blob, dim = r
        vec = _from_blob(blob, dim)
        ids.append(kid)
        vectors.append(vec)

    arr = np.vstack(vectors).astype('float32')
    _dim = arr.shape[1]
    # normalizează numerele pentru a asigura că sistemul calculează corect distanța (similitudinea) dintre întrebare și răspunsuri.
    faiss.normalize_L2(arr)

    idx = faiss.IndexFlatIP(_dim)
    idx.add(arr)
    _index = idx
    _id_map = np.array(ids, dtype=int)

    try:
        faiss.write_index(_index, FAISS_INDEX_PATH)
        np.save(FAISS_MAP_PATH, _id_map)
    except Exception as e:
        print("Warning: could not save FAISS index:", e)
# Când pornește aplicația, verifică dacă există deja un index salvat pe disc pentru a nu-l reconstrui de la zero.
def load_faiss_index_if_exists():
    global _index, _id_map, _dim
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAP_PATH):
            idx = faiss.read_index(FAISS_INDEX_PATH)
            ids = np.load(FAISS_MAP_PATH, allow_pickle=True)
            _index = idx
            _id_map = ids.astype(int)
            _dim = idx.d
            return True
    except Exception as e:
        print("Could not load saved FAISS index:", e)
    return False

def search_semantic(query: str, top_k=3):
    global _index, _id_map
    if _index is None:
        return []

    q_emb = compute_embedding(query)
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = _index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        knowledge_id = int(_id_map[idx])
        con = get_connection()
        cur = con.cursor()
        cur.execute("SELECT content, subject, grade FROM knowledge WHERE id=?", (knowledge_id,))
        row = cur.fetchone()
        con.close()
        if row:
            content, subject, grade = row
            results.append({"id": knowledge_id, "score": float(score), "content": content, "subject": subject, "grade": grade})
    return results

def ensure_index():
    if not load_faiss_index_if_exists():
        build_faiss_index()
