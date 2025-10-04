short_term_memory = {}

def get_short_term_memory():
    return short_term_memory

def update_short_term_memory(fact, embedding):
    short_term_memory[fact] = embedding
