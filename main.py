# main.py
from graph.main_graph import graph
from langchain_core.messages import HumanMessage

# This is your persistent, fixed thread ID
CONVERSATION_ID = "karthi_main_session"

def run_ai(query, conversation_id):
    """
    Runs the agentic graph using a streaming loop to get the final response.
    """
    config = {"configurable": {"thread_id": conversation_id}}
    
    # 1. Define the input
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # 2. Use .stream() to run the graph and get chunks
    # This loop will run until the agent hits the "respond" node.
    final_response = None
    for chunk in graph.stream(inputs, config=config):
        # The 'chunk' is the output of the *last node* that ran
        # We only care about the final response from the "respond" node
        if "respond" in chunk:
            # The 'respond_and_save_node' prints the message,
            # so we just break the loop.
            break

if __name__ == "__main__":
    print("🤖 JARVIS: Online. (Loading persistent memory...)")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        # This loop now correctly waits for the agent to finish
        # before asking for new input.
        run_ai(user_query, CONVERSATION_ID)