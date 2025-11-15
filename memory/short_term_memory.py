#short_term_memory.py
short_term_memory = {}

def get_short_term_memory():
    return short_term_memory

def update_short_term_memory(fact, embedding):
    short_term_memory[fact] = embedding

def get_short_term_memory_facts():
    """Returns a list of facts stored in short-term memory."""
    return list(short_term_memory.keys())
