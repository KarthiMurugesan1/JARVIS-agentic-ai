import requests
from utils.config import PERPLEXITY_API_KEY

def perplexity_search(query):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}"}
    data = {"model": "pplx-7b-online", "messages": [{"role": "user", "content": query}]}
    response = requests.post(url, json=data, headers=headers).json()
    return response.get("output", "No result found")
