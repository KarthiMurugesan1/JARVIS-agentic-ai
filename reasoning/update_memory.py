from memory.short_term_memory import update_short_term_memory
from memory.long_term_memory import update_memory as update_long_term_memory

def update_memory(fact, embedding, storage_type="long"):
    """
    Update memory based on LLM decision
    storage_type: "short" or "long"
    """
    if storage_type == "short":
        update_short_term_memory(fact, embedding)
    else:
        update_long_term_memory(fact, embedding)
