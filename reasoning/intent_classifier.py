def classify_intent(query):
    query_lower = query.lower()
    if "search" in query_lower or "find" in query_lower or "look up" in query_lower:
        return "web_search"
    elif "my" in query_lower or "i" in query_lower or "remember" in query_lower:
        return "user_info"
    else:
        return "general"
