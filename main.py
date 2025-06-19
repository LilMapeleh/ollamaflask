from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
CORS(app)

# Langchain/Ollama setup
template = """
You are a helpful and friendly AI product assistant helping users design software solutions.
Your job is to guide them through understanding their ideas, refining their needs, and generating user stories and requirements.

Current Phase: {phase}
Conversation History: {context}
User Input: {question}

If the user says something short or vague (like "hi", "hello", "yes"), encourage them to share a solution idea they want built.
Avoid repeating yourself. Ask specific follow-up questions to guide the user from idea to story.

Respond clearly and naturally:
- If first message: welcome and ask what system they want built
- If unclear input: ask clarifying questions
- If detailed input: help shape it into user stories

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

conversation_state = {
    "context": "",
    "phase": "Business Problem"
}

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    # Return initial bot message if triggered by frontend
    if user_input == "__INIT__":
        initial_message = (
          "👋 Hello! I'm REA, Rangka Empat's AI business assistant. I'm here to help you shape and submit your idea to the Team."
        "Just tell me what kind of IT solution you’re thinking about — like a booking system, inventory tracker, or customer feedback tool — and I’ll help you:\n\n"
        "• Refine your idea\n"
        "• Propose a suitable solution\n"
        "• Define clear features and user requirements\n"
        "• Generate helpful user stories for the Team\n\n"
        "Ready to begin? Just describe the kind of system you need, and we’ll build it together step by step!"
    )
        conversation_state["context"] = f"AI: {initial_message}"
        return jsonify({"reply": initial_message, "phase": conversation_state["phase"]})

    # Pull from context
    context = conversation_state["context"]
    phase = conversation_state["phase"]

    # Langchain invocation
    result = chain.invoke({
        "context": context,
        "question": user_input,
        "phase": phase
    })

    ai_reply = getattr(result, "text", str(result))

    # Update memory
    context += f"\nUser: {user_input}\nAI: {ai_reply}"
    conversation_state["context"] = context

    # Phase logic
    if "solution idea" in context.lower() and phase == "Business Problem":
        conversation_state["phase"] = "Solution Idea"
    elif "feature" in context.lower() and phase == "Solution Idea":
        conversation_state["phase"] = "Feature Requirements"
    elif "user story" in context.lower() and phase == "Feature Requirements":
        conversation_state["phase"] = "User Stories"

    return jsonify({"reply": ai_reply, "phase": conversation_state["phase"]})

if __name__ == "__main__":
    app.run(debug=True)
