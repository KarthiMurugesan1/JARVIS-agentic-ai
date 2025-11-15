# reasoning/llm_reasoning.py
from perception.perplexity_api import perplexity_search

def llm_reasoning(query, context):
    """
    (Original function)
    Combine query + retrieved memory or web search context,
    and generate the final answer via the Perplexity API.
    """
    if context:
        prompt = f"Context: {context}\n\nUser Query: {query}"
    else:
        prompt = query
    response = perplexity_search(prompt)
    return response

# --- NEW FUNCTION for our Looping Agent ---
def llm_reasoning_with_history(system_prompt: str, history: list):
    """
    Generates a response using the Perplexity API with a full
    conversation history.
    """
    
    # Convert LangChain messages to a simple string format
    # that Perplexity can understand.
    formatted_history = []
    for msg in history:
        if msg.type == "human":
            formatted_history.append(f"User: {msg.content}")
        elif msg.type == "ai":
            if msg.tool_calls:
                # Get the tool call info
                tool_call = msg.tool_calls[0]
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                formatted_history.append(f"JARVIS: (Calling tool: {tool_name} with args: {tool_args})")
            else:
                formatted_history.append(f"JARVIS: {msg.content}")
        elif msg.type == "tool":
            formatted_history.append(f"Tool Result: {msg.content}")
            
    # Combine the system prompt with the history
    full_prompt = system_prompt + "\n\n**Conversation History:**\n" + "\n".join(formatted_history)
    
    # Add a final prompt for JARVIS to respond
    full_prompt += "\n\nJARVIS: "
    
    # Call the Perplexity API
    response = perplexity_search(full_prompt)
    
    return response.strip()