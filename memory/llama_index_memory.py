# memory/llama_index_memory.py

import os
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from .local_embedding import LocalEmbedding

# -----------------------
# Load DATABASE_URL
# -----------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing! Must be set in Render environment variables.")

# -----------------------
# Embedding model (CPU)
# -----------------------
embed_model = LocalEmbedding()

# -----------------------
# Initialize PGVectorStore using DATABASE_URL
# -----------------------
def get_pg_vector_store():
    """
    Initialize PostgreSQL pgvector store using DATABASE_URL directly.
    Compatible with Render Postgres.
    """
    return PGVectorStore.from_params(
        connection_string=DATABASE_URL,
        table_name="llamaindex_memory",
        embed_dim=384  # all-MiniLM-L6-v2 embedding dimension
    )

# -----------------------
# Get or create index
# -----------------------
def get_index():
    try:
        vector_store = get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    except Exception as e:
        print(f"[LlamaIndex] Error loading index: {e}")
        print("[LlamaIndex] Creating new index...")

        vector_store = get_pg_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)

# -----------------------
# Store memory
# -----------------------
def store_memory(summary: str):
    index = get_index()
    doc = Document(text=summary)
    index.insert(doc)

# -----------------------
# Retrieve memory
# -----------------------
def retrieve_relevant_memory(query: str, top_k=5):
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    docs = retriever.retrieve(query)
    return [d.text for d in docs]
