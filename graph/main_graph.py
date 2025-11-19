# graph/main_graph.py

import os
import json
import re
import operator
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
load_dotenv()

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

# Internal modules
from perception.perplexity_api import perplexity_search
from reasoning.llm_reasoning import llm_reasoning_with_history
from memory.short_term_memory import update_short_term_memory
from memory.local_embedding import get_embedding
from tools.tool_registry import AVAILABLE_TOOLS, TOOL_DESCRIPTIONS
from utils.config import POSTGRES_CONFIG


# ============================================================
# =============== LOAD ENVIRONMENT VARIABLES =================
# ============================================================

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
STM_CONDENSE_THRESHOLD = int(os.getenv("STM_CONDENSE_THRESHOLD", 5))


# ============================================================
# ===== DATABASE_URL HANDLING (LOCAL + RENDER COMPATIBLE) =====
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL or DATABASE_URL.strip() == "":
    print("⚠️ DATABASE_URL not set. Using local PostgreSQL from .env ...")

    DATABASE_URL = (
        f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@"
        f"{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/"
        f"{POSTGRES_CONFIG['dbname']}"
    )

    print(f"✔️ Local DATABASE_URL = {DATABASE_URL}")


# ============================================================
# ======================= AGENT STATE =========================
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================
# ===================== PLANNER NODE =========================
# ============================================================

def call_planner_llm(state: AgentState):
    print("🤖 [Node] Planner LLM is thinking...")
    messages = state["messages"]

    system_prompt = f"""
You are JARVIS, a proactive, highly capable AI assistant.

You have access to these tools:
{TOOL_DESCRIPTIONS}

Rules:
1. If last message is a ToolMessage → summarize result for user.
2. If HumanMessage:
   - Use memory tools for personal info.
   - Use system tools for system actions.
   - Use search_web for general knowledge.
   - Otherwise answer normally.
3. If planning to use a tool → output ONLY JSON tool call.
4. Otherwise → output ONLY the answer.
"""

    llm_response = llm_reasoning_with_history(system_prompt, messages)

    try:
        match = re.search(r"\{.*\}", llm_response, re.DOTALL)

        if match:
            tool_call_json = json.loads(match.group())
            tool_name = tool_call_json.get("tool_name")
            parameters = tool_call_json.get("parameters", {})

            print(f"🤖 [Planner] Calling tool: {tool_name}")

            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "id": f"tool_{len(messages)}",
                            "name": tool_name,
                            "args": parameters
                        }]
                    )
                ]
            }
        else:
            print("🤖 [Planner] Responding directly.")
            return {"messages": [AIMessage(content=llm_response)]}

    except Exception as e:
        print(f"Planner error: {e}, LLM said: {llm_response}")
        return {"messages": [AIMessage(content="I got confused. Please try again.")]}


# ============================================================
# ==================== TOOL EXECUTOR NODE ====================
# ============================================================

def call_tool_executor(state: AgentState):
    print("🤖 [Node] Tool Executor")

    last = state["messages"][-1]
    tool_call = last.tool_calls[0]

    name = tool_call["name"]
    args = tool_call["args"]

    if name not in AVAILABLE_TOOLS:
        return {
            "messages": [
                ToolMessage(content=f"Error: Tool '{name}' not found.", tool_call_id=tool_call["id"])
            ]
        }

    try:
        result = AVAILABLE_TOOLS[name](**args)
        print(f"🤖 [Tool Executor] {name} succeeded")
        return {
            "messages": [
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            ]
        }
    except Exception as e:
        print(f"Error running tool {name}: {e}")
        return {
            "messages": [
                ToolMessage(content=f"Error running tool: {e}", tool_call_id=tool_call["id"])
            ]
        }


# ============================================================
# =================== FINAL RESPONSE NODE ====================
# ============================================================

def respond_and_save_node(state: AgentState):
    final_response = state["messages"][-1].content
    print("🤖 JARVIS:", final_response)

    # find last user message
    user_query = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    summary = f'User: "{user_query}" | JARVIS: "{final_response}"'

    decision_prompt = f"""
Memory filter:
"{summary}"

Return ONLY: SAVE or IGNORE.
"""

    try:
        decision = perplexity_search(decision_prompt).strip().upper()
    except:
        decision = "SAVE"

    if decision == "SAVE":
        embedding = get_embedding(summary)
        update_short_term_memory(summary, embedding)
        print(f"[STM] Saved: {summary}")
    else:
        print("[STM] Ignored trivial exchange.")

    return state


# ============================================================
# ======================== ROUTER =============================
# ============================================================

def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "call_tool" if last.tool_calls else "end"


# ============================================================
# ======================= BUILD GRAPH =========================
# ============================================================

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
        "end": "respond"
    }
)

graph_builder.add_edge("tool_executor", "planner_llm")
graph_builder.add_edge("respond", END)


# ============================================================
# =============== CHECKPOINT (POSTGRES + SQLITE) ==============
# ============================================================

# IMPORTANT: langgraph-checkpoint-sqlite v3.x uses new signature.
memory = SqliteSaver.from_conn_string(DATABASE_URL)

graph = graph_builder.compile(checkpointer=memory)
