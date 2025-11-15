from perception.perplexity_api import perplexity_search
import re
import json
# Import the tool descriptions so the classifier knows what's possible
from tools.tool_registry import TOOL_DESCRIPTIONS

def classify_intent(query: str, retrieved_memory: str = None):
    prompt = f"""
You are JARVIS, an intelligent AI assistant.
Your goal is to classify the user's intent.
User input: "{query}"
Memory context: "{retrieved_memory or 'No memory yet.'}"

Here are the available tools you can use:
{TOOL_DESCRIPTIONS}

Classify the user's intent into one of the following:
- "NEW_FACT": User is providing new info to remember (e.g., "my name is...", "my favorite color is...").
- "MEMORY_QUERY": User is asking about info that might exist in memory (e.g., "what is my name?", "what is my favorite color?").
- "TOOL_USE": The user's request can be answered by using one of the available tools (e.g., "play a song", "what is the weather", "who won the game").
- "GENERAL": Normal conversation or a request you cannot fulfill (e.g., "hello", "how are you?", "what can you do?").

Return JSON like:
{{
    "intent": "<NEW_FACT|MEMORY_QUERY|TOOL_USE|GENERAL>"
}}
"""
    response = perplexity_search(prompt)
    match = re.search(r'\{.*\}', response, re.DOTALL) # Added re.DOTALL
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"intent": "GENERAL"}
    return {"intent": "GENERAL"}