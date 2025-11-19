import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Assuming 'graph' and 'HumanMessage' are correctly imported from your modules
from graph.main_graph import graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="JARVIS Agentic AI API", version="1.0.0")

# 1. Define the input data structure for the API
class ChatQuery(BaseModel):
    # Use a session_id instead of a fixed CONVERSATION_ID for multi-user support
    session_id: str
    query: str

def get_final_response(query: str, conversation_id: str) -> Optional[str]:
    """
    Runs the agentic graph, returning only the final response.
    """
    # **IMPORTANT:** Use the session_id as the thread_id for state management
    config = {"configurable": {"thread_id": conversation_id}}
    inputs = {"messages": [HumanMessage(content=query)]}
    
    final_message = None
    
    # Use .stream() to run the graph and get chunks
    for chunk in graph.stream(inputs, config=config):
        # We assume the 'respond' node contains the final output message
        if "respond" in chunk:
            # Adjust this line based on the exact structure of your final output
            # This is a common pattern for LangGraph/LangChain runnables
            final_message = chunk["respond"]["messages"][-1].content 
            break
            
    return final_message or "Agent finished, but did not return a response."


@app.post("/chat")
async def chat_endpoint(data: ChatQuery) -> Dict[str, Any]:
    """
    Primary endpoint for interacting with the agent.
    """
    print(f"Received query from session {data.session_id}: {data.query}")
    
    response_text = get_final_response(data.query, data.session_id)
    
    return {
        "session_id": data.session_id,
        "response": response_text,
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "JARVIS-agentic-ai"}