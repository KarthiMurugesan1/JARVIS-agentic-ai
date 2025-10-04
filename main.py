from graph.main_graph import graph

def run_ai(query):
    state = {"query": query, "intent": "", "context": "", "response": ""}
    final_state = graph.invoke(state)
    return final_state["response"]

if __name__ == "__main__":
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = run_ai(user_query)
        print("🤖 JARVIS:", response)
