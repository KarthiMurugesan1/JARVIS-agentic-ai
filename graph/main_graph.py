# graph/main_graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
import operator
import json
import re
from perception.perplexity_api import perplexity_search

# We will use LangChain's message types to store history
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

# ---- Imports for reasoning, memory, and TOOLS ---- #
from reasoning.llm_reasoning import llm_reasoning_with_history
from memory.short_term_memory import update_short_term_memory
from memory.local_embedding import get_embedding
from tools.tool_registry import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS

# ---- 1. Define the State Schema ---- #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ---- 2. Define the Nodes ---- #

def call_planner_llm(state: AgentState):
    """
    This is the "brain." It looks at the conversation history and
    decides what to do next: call a tool or respond to the user.
    """
    print("🤖 [Node] Planner LLM is thinking...")
    messages = state['messages']
    
    # --- THIS IS THE UPGRADED, CLEARER PROMPT ---
    system_prompt = f"""
You are JARVIS, a proactive and personalized AI assistant.
Your job is to analyze the conversation and the result of any tool calls, then decide on the next logical step.
You have access to the following tools:
{TOOL_DESCRIPTIONS}

**Your Core Principles (In Order):**

1.  **Analyze the Last Message:**
    * **If the last message is a `ToolMessage` (a result from a tool):** Your *only* job is to synthesize this new information into a clear, natural language answer for the user.
    * **If the last message is a `HumanMessage` (a new query):** You must decide what to do.

2.  **How to Decide on a Plan (for Human Messages):**
    * **Personal Memory:** For questions about the user (e.g., "what's my name?") or saving new facts (e.g., "my name is..."), use the `retrieve_memory` or `save_memory` tools.
    * **System Actions:** For requests to find/open files, play music, or check system stats, use the appropriate tool (`find_file`, `open_path`, `play_song_on_youtube`, `get_system_stats`).
    * **General Knowledge:** For *any* other question about the world, facts, news, or people (e.g., "who is ceo of tata"), you *must* use the `search_web` tool.
    * **Simple Chat:** For simple greetings or chat (e.g., "hello", "how are you"), just respond naturally.

3.  **Re-planning Logic:**
    * If `open_path` fails, your next step *must* be `find_file`.
    * If `find_file` succeeds, your next step *must* be `open_path` with the new path.
    * If `find_file` fails, you *must* stop and report the failure to the user.

4.  **Final Output:**
    * If your plan is to use a tool, return *only* the JSON for that tool call.
    * If your plan is to respond to the user, return *only* that natural language response.

**Tool Call JSON Format:**
{{
    "tool_name": "name_of_the_tool",
    "parameters": {{"arg_name": "arg_value"}}
}}
"""
    # --- END OF PROMPT UPGRADE ---
    
    llm_response = llm_reasoning_with_history(system_prompt, messages)
    
    try:
        match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if match:
            tool_call_json = json.loads(match.group())
            tool_name = tool_call_json.get("tool_name")
            parameters = tool_call_json.get("parameters", {})
            
            print(f"🤖 [Node] Planner wants to call tool: {tool_name}")
            new_ai_message = AIMessage(
                content="",
                tool_calls=[{
                    "id": f"tool_{len(messages)}",
                    "name": tool_name,
                    "args": parameters
                }]
            )
        else:
            print("🤖 [Node] Planner is responding directly.")
            new_ai_message = AIMessage(content=llm_response)
    
    except Exception as e:
        print(f"Error in planner: {e}. LLM response was: {llm_response}")
        new_ai_message = AIMessage(content="Sorry, I got confused. Please try again.")

    return {"messages": [new_ai_message]}


def call_tool_executor(state: AgentState):
    """
    This node executes the tool call requested by the planner LLM.
    """
    print("🤖 [Node] Tool Executor")
    last_message = state['messages'][-1]
    
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    parameters = tool_call["args"]
    
    if tool_name in AVAILABLE_TOOLS:
        tool_function = AVAILABLE_TOOLS[tool_name]
        try:
            tool_result = tool_function(**parameters)
            tool_result_str = str(tool_result)
            print(f"🤖 [Node] Tool '{tool_name}' succeeded.")
        except Exception as e:
            tool_result_str = f"Error running tool: {e}"
            print(f"Error calling tool {tool_name}: {e}")
    else:
        tool_result_str = f"Error: Tool '{tool_name}' does not exist."

    return {"messages": [ToolMessage(content=tool_result_str, tool_call_id=tool_call["id"])]}


def respond_and_save_node(state: AgentState):
    """
    This node prints the final response and *conditionally* saves it to STM.
    """
    final_response = state['messages'][-1].content
    print("🤖 JARVIS:", final_response)
    
    # --- Find the last user query ---
    user_query = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
            
    summary = f"User: \"{user_query}\" | JARVIS: \"{final_response}\""

    # --- New STM Decision Logic ---
    # We ask the LLM if this summary is worth saving
    decision_prompt = f"""
You are a memory filter. A conversation just ended.
Summary: "{summary}"

Is this exchange trivial (e.g., "hi", "how are you", "thanks", "ok", "you're welcome")?
Or does it contain new information, a question, a plan, or a meaningful interaction?

Answer with a single word: 'SAVE' or 'IGNORE'.
"""
    
    try:
        # We use the raw perplexity_search for a simple, non-history call
        decision = perplexity_search(decision_prompt).strip().upper()
    except Exception as e:
        print(f"[STM] Error during decision: {e}")
        decision = "SAVE" # Default to saving if decision fails

    if "SAVE" in decision:
        embedding = get_embedding(summary)
        update_short_term_memory(summary, embedding)
        print(f"[STM] Stored: {summary}")
    else:
        # This is what you wanted
        print(f"[STM] Ignored trivial exchange.")
    # --- End of New Logic ---
    
    return state

# ---- 3. Define the Graph Edges (The Router) ---- #

def should_continue(state: AgentState):
    """
    This is the "router" that creates the loop.
    """
    last_message = state['messages'][-1]
    
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "end"

# ---- 4. Build the Graph ---- #
graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner_llm", call_planner_llm)
graph_builder.add_node("tool_executor", call_tool_executor)
graph_builder.add_node("respond", respond_and_save_node)

graph_builder.set_entry_point("planner_llm")

graph_builder.add_conditional_edges(
    "planner_llm",
    should_continue,
    {
        "call_tool": "tool_executor",
        "end": "respond",
    },
)

graph_builder.add_edge("tool_executor", "planner_llm")
graph_builder.add_edge("respond", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)