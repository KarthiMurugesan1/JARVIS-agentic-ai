# In JARVIS-agentic-ai/memory/local_embedding.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

# Create one global model instance
_embed_model_instance = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)
print("Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2")

class LocalEmbedding(BaseEmbedding):
    _embed_model: HuggingFaceEmbedding = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._embed_model = _embed_model_instance # Use the global instance

    # ... (rest of the class methods) ...
    def _get_text_embedding(self, text: str):
        return self._embed_model.get_text_embedding(text)

    def _get_query_embedding(self, query: str):
        return self._embed_model.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

# Add this function for the graph to use
def get_embedding(text: str):
    """Get embedding for text using the global model."""
    if isinstance(text, str):
        return _embed_model_instance.get_text_embedding(text)
    elif isinstance(text, list):
        return _embed_model_instance.get_text_embeddings(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")