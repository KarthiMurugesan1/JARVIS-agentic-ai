from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any

from reasoning.intent_classifier import classify_intent
from memory.long_term_memory import retrieve_memory
from perception.perplexity_api import perplexity_search
from reasoning.llm_reasoning import llm_reasoning
from reasoning.update_memory import update_memory

# ---- Define the State Schema ---- #
class AgentState(TypedDict):
    query: str
    intent: str
    context: Any
    response: str

# ---- Initialize Graph ---- #
graph = StateGraph(AgentState)

# ---- Define Nodes ---- #
def classify_intent_node(state: AgentState):
    intent = classify_intent(state["query"])
    state["intent"] = intent
    return state

def fetch_user_memory_node(state: AgentState):
    # You would normally compute embedding for the query here
    query_embedding = [0.1, 0.2, 0.3]  # placeholder
    results = retrieve_memory(query_embedding)
    state["context"] = results
    return state

def perplexity_search_node(state: AgentState):
    results = perplexity_search(state["query"])
    state["context"] = results
    return state

def llm_reasoning_node(state: AgentState):
    output = llm_reasoning(state["query"], state.get("context", ""))
    state["response"] = output
    return state

def update_memory_node(state: AgentState):
    # Optional: run update logic only for non-web_search intents
    if state["intent"] != "web_search":
        update_memory(state["query"], [0.1, 0.2, 0.3], "long")
    return state

def respond_node(state: AgentState):
    print("🤖 JARVIS:", state["response"])
    return state

# ---- Add Nodes ---- #
graph.add_node("classify_intent", classify_intent_node)
graph.add_node("fetch_user_memory", fetch_user_memory_node)
graph.add_node("perplexity_search", perplexity_search_node)
graph.add_node("llm_reasoning", llm_reasoning_node)
graph.add_node("update_memory", update_memory_node)
graph.add_node("respond", respond_node)

# ---- Connect Nodes ---- #
graph.add_edge(START, "classify_intent")

# Intent-based branching
graph.add_conditional_edges(
    "classify_intent",
    lambda state: state["intent"],
    {
        "user_info": "fetch_user_memory",
        "web_search": "perplexity_search",
        "general": "llm_reasoning"
    },
)

# Merge paths
graph.add_edge("fetch_user_memory", "llm_reasoning")
graph.add_edge("perplexity_search", "llm_reasoning")

# Update memory only if intent != web_search
graph.add_conditional_edges(
    "llm_reasoning",
    lambda state: state["intent"],
    {
        "web_search": "respond",
        "user_info": "update_memory",
        "general": "update_memory"
    },
)

graph.add_edge("update_memory", "respond")
graph.add_edge("respond", END)

# ---- Compile ---- #
graph = graph.compile()
