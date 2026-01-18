import sqlite3
from config import DB_PATH

def get_connection():
    con = sqlite3.connect(DB_PATH)
    con.text_factory = str
    return con
