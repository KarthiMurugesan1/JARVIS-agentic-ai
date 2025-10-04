import psycopg2
from utils.config import POSTGRES_CONFIG

def get_connection():
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    return conn

def retrieve_memory(query_embedding, top_k=5):
    conn = get_connection()
    cur = conn.cursor()
    # Use pgvector for similarity search
    cur.execute("""
        SELECT fact
        FROM long_term_memory
        ORDER BY embedding <-> %s
        LIMIT %s;
    """, (query_embedding, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in results]

def update_memory(fact, embedding):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO long_term_memory (fact, embedding, source) VALUES (%s, %s, %s)",
        (fact, embedding, "user")
    )
    conn.commit()
    cur.close()
    conn.close()
