# perception/perplexity_api.py
import requests
from utils.config import PERPLEXITY_API_KEY

def perplexity_search(query, context=None):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # ✅ Merge context and query into a single user message
    if context:
        user_prompt = f"Use the following context to answer:\n{context}\n\nUser query: {query}"
    else:
        user_prompt = query

    data = {
        "model": "sonar",  # or sonar-medium-chat / sonar
        "messages": [
            {"role": "system", "content": "You are JARVIS, a helpful and intelligent AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()

    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"Error: {response_json}"
