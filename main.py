from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Flask app setup
app = Flask(__name__)
CORS(app)

# Langchain/Ollama setup
template = """
You are a business solution assistant. Your goal is to help the user identify their business problem, refine the solution, gather requirements, and generate user stories.

Current Phase: {phase}
Conversation History: {context}
User Input: {question}

Respond appropriately for this phase by asking relevant questions or providing output like user stories.

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Store context and phase in memory (reset every time server restarts)
conversation_state = {
    "context": "",
    "phase": "Business Problem"
}

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    # Use global conversation state
    context = conversation_state["context"]
    phase = conversation_state["phase"]

    result = chain.invoke({
        "context": context,
        "question": user_input,
        "phase": phase
    })

    # Update context
    context += f"\nUser: {user_input}\nAI: {result}"
    conversation_state["context"] = context

    # Rule-based phase switching
    if "business problem" in context.lower() and phase == "Business Problem":
        conversation_state["phase"] = "Solution Idea"
    elif "solution" in context.lower() and phase == "Solution Idea":
        conversation_state["phase"] = "Feature Requirements"
    elif "features" in context.lower() and phase == "Feature Requirements":
        conversation_state["phase"] = "User Stories"

    return jsonify({"reply": result, "phase": conversation_state["phase"]})

if __name__ == "__main__":
    app.run(debug=True)
