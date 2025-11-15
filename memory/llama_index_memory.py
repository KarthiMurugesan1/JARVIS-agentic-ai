# memory/llama_index_memory.py

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from utils.config import POSTGRES_CONFIG
from .local_embedding import LocalEmbedding

# Initialize local embedding model
embed_model = LocalEmbedding()

def get_pg_vector_store():
    """
    Initialize PostgreSQL-backed vector store dynamically and safely.
    Handles empty passwords, missing port, and default values.
    """
    # Read ALL connection info from the config
    user = POSTGRES_CONFIG.get("user", "karthimurugesan")
    password = POSTGRES_CONFIG.get("password")  # Can be None or empty string
    host = POSTGRES_CONFIG.get("host", "localhost")
    port = POSTGRES_CONFIG.get("port", 5432)
    dbname = POSTGRES_CONFIG.get("dbname", "JARVIS")

    if password:
        conn_str_debug = f"postgresql+asyncpg://{user}:***@{host}:{port}/{dbname}"
    else:
        conn_str_debug = f"postgresql+asyncpg://{user}@{host}:{port}/{dbname}"
    print(f"[DEBUG] Config read, attempting to connect with: {conn_str_debug}")


    # Pass parameters individually to from_params
    return PGVectorStore.from_params(
        user=user,
        password=password,
        host=host,
        port=port,
        database=dbname,
        table_name="llamaindex_memory",
        embed_dim=384  # <-- THIS IS THE FIX
    )


def get_index():
    """
    Get LlamaIndex instance.
    Creates a new index if it does not exist or fails to load.
    """
    try:
        vector_store = get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    except Exception as e:
        # If the table exists with the wrong dimensions, we might need to recreate it.
        # This simple logic assumes we can just try again.
        print(f"[LlamaIndex] Error getting index (might be dimension mismatch): {e}")
        print("[LlamaIndex] Trying to create new index...")
        vector_store = get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # This will create the table if it doesn't exist with the right dimensions
        return VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)


def store_memory(summary: str):
    """
    Store a summary string into LlamaIndex (Postgres).
    Automatically handles document creation and insertion.
    """
    index = get_index()
    doc = Document(text=summary)
    index.insert(doc)


def retrieve_relevant_memory(query: str, top_k=5):
    """
    Retrieve top-k relevant memory entries for a given query.
    Returns a list of memory text strings.
    """
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    docs = retriever.retrieve(query)
    return [d.text for d in docs]