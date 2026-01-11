from database.db import get_connection

def init():
    con = get_connection()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        grade TEXT,
        content TEXT
    )
    """)
 # tabel în care salvăm embedding-urile ca BLOB (și dimensiunea)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,  -- id = knowledge.id
        vector BLOB,
        dim INTEGER
    )
    """)

    con.commit()
    con.close()
    print("Database initialized (knowledge + embeddings).")

if __name__ == '__main__':
    init()
