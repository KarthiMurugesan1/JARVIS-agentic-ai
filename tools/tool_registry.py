# tools/tool_registry.py
import webbrowser
import re
import os
import subprocess
import shlex
import psutil
import difflib
from perception.perplexity_api import perplexity_search
from memory.llama_index_memory import store_memory, retrieve_relevant_memory

# --- Tool 1: Web Search ---
def search_web(query: str):
    """search_web(query: str): Searches the web for an answer to a user's query. Use this for facts, news, or general knowledge."""
    print(f"🤖 [Tool] Searching web for: {query}")
    return perplexity_search(query)

# --- Tool 2: Play Music ---
def play_song_on_youtube(song_query: str):
    """play_song_on_youtube(song_query: str): Opens a YouTube search in a web browser for the requested song. Use this when a user asks to play a song."""
    search_query = re.sub(r'\s+', '+', song_query)
    url = f"https://www.youtube.com/results?search_query={search_query}"
    try:
        print(f"🤖 [Tool] Opening browser to search for: {song_query}")
        webbrowser.open(url)
        return f"Successfully opened YouTube search for '{song_query}'."
    except Exception as e:
        return f"Sorry, I couldn't open the browser: {e}"

# --- Tool 3: List Files ---
def list_files(directory: str = "."):
    """list_files(directory: str): Lists files and folders in a specified directory. The default is the current directory ('.')."""
    print(f"🤖 [Tool] Listing files in: {directory}")
    safe_path = os.path.realpath(os.path.expanduser(directory))
    if not os.path.isdir(safe_path):
        return f"Error: '{directory}' is not a valid directory."
    try:
        files = os.listdir(safe_path)
        return f"Contents of '{directory}':\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"

# --- Tool 4: Open Path ---
def open_path(path: str):
    """open_path(path: str): Opens a specific file or directory using the system's default application (e.g., '~/Downloads/file.pdf')."""
    print(f"🤖 [Tool] Opening path: {path}")
    safe_path = os.path.realpath(os.path.expanduser(path))
    if not os.path.exists(safe_path):
        return f"Error: Path '{path}' does not exist."
    try:
        # Use file:// URI scheme to ensure browser opens local files
        path_uri = f"file://{safe_path}"
        webbrowser.open(path_uri)
        return f"Successfully opened '{path}'."
    except Exception as e:
        return f"Error opening path: {e}"

# --- Tool 5: Get System Stats ---
def get_system_stats():
    """get_system_stats(): Gets system RAM and disk space."""
    print("🤖 [Tool] Getting system stats")
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return f"RAM: {mem.available / (1024**3):.2f}GB available. Disk: {disk.free / (1024**3):.2f}GB free."

# --- Tool 6: Find File ---
def find_file(filename: str):
    """find_file(filename: str): Searches the user's home directory for a file by name. Returns the path of the closest match."""
    print(f"🤖 [Tool] Searching for file: {filename}")
    start_dir = os.path.expanduser('~')
    matches = []
    
    # Folders to skip for speed
    skip_dirs = {
        'Library', 'Application Support', 'node_modules', '.git', 
        '.cache', '.venv', 'venv', 'anaconda3', 'miniconda3',
        'Applications', 'System', 'Pictures', 'Music', 'Movies', 'Public'
    }

    for root, dirs, files in os.walk(start_dir, topdown=True):
        # Prune search by removing skipped directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        
        for file in files:
            if filename.lower() in file.lower():
                matches.append(os.path.join(root, file))
            # Limit matches to avoid searching forever
            if len(matches) > 100:
                break
        if len(matches) > 100:
            break

    if not matches:
        return "Error: No file found matching that name."

    # Find the best match using text similarity
    best_match = max(matches, key=lambda x: difflib.SequenceMatcher(None, filename, os.path.basename(x)).ratio())
    print(f"Found best match: {best_match}")
    return f"Found file at: {best_match}"

def save_memory(fact: str):
    """save_memory(fact: str): Saves a personal fact, user preference, or important detail to long-term memory. Use this when the user states a new piece of information about themselves (e.g., "my name is...", "my favorite color is blue")."""
    print(f"🤖 [Tool] Saving to LTM: {fact}")
    try:
        store_memory(fact)  # This calls your LlamaIndex function
        return "Successfully saved fact to long-term memory."
    except Exception as e:
        print(f"Error saving memory: {e}")
        return "Error: Could not save fact to memory."
    
def retrieve_memory(query: str):
    """retrieve_memory(query: str): Retrieves relevant facts from long-term memory based on a query. Use this *first* for any personal questions (e.g., "what is my name?")."""
    print(f"🤖 [Tool] Retrieving from LTM for query: {query}")
    try:
        results = retrieve_relevant_memory(query, top_k=3)
        if not results:
            return "No relevant information found in long-term memory."
        return f"Found relevant facts in memory: {'; '.join(results)}"
    except Exception as e:
        print(f"Error retrieving memory: {e}")
        return "Error: Could not retrieve facts from memory."


# --- The Tool Registry ---
AVAILABLE_TOOLS = {
    "search_web": search_web,
    "play_song_on_youtube": play_song_on_youtube,
    "list_files": list_files,
    "open_path": open_path,
    "get_system_stats": get_system_stats,
    "find_file": find_file,
    "save_memory": save_memory,  # <-- 2. ADD THE TOOL TO THE DICTIONARY
    "retrieve_memory": retrieve_memory,
}

# --- DYNAMIC PROMPT GENERATION ---
def generate_tool_descriptions():
    """Reads the docstring of each tool in AVAILABLE_TOOLS to build the prompt."""
    descriptions = []
    for func in AVAILABLE_TOOLS.values():
        if func.__doc__:
            doc = func.__doc__.strip()
            descriptions.append(f'- "{doc}"')
    return "\n".join(descriptions)

# This variable is now *dynamically generated*
TOOL_DESCRIPTIONS = generate_tool_descriptions()