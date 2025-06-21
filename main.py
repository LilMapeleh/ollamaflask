from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from send_email import send_email
import re

app = Flask(__name__)
CORS(app)

# Langchain/Ollama setup
template = """
You are a helpful AI assistant helping users define software projects.
Your job is to understand the user's business problem and proposed solution (if provided),
then guide them through these stages:

1. Understand their business problem
2. Capture their proposed solution (unless they already gave it)
3. Ask logical questions to extract required features
4. Confirm features
5. Generate user stories
6. Confirm again
7. Ask for name & phone
8. Offer to send to dev team

Use this conversation history and question to decide what to do next.

Current Phase: {phase}
Conversation History: {context}
User Input: {question}

Respond clearly and naturally. If the user has already proposed a solution, move to asking business logic questions needed to define requirements.
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

conversation_state = {
    "context": "",
    "phase": "Business Problem",
    "awaiting_confirmation": False,
    "user_details": {},
    "email_ready": False,
    "skipped_solution": False
}

def is_solution_statement(text):
    keywords = ["build", "create", "develop", "mobile app", "website", "system", "platform", "portal", "dashboard"]
    return any(kw in text.lower() for kw in keywords)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    # Init message
    if user_input == "__INIT__":
        initial_message = (
            "👋 Hello! I'm REA, your assistant from Rangka Empat.\n\n"
            "Tell me what kind of IT system you're thinking about — like a booking system, inventory tracker, or feedback tool.\n"
            "I'll help you shape your idea, define clear features, and generate user stories ready for development.\n\n"
            "Let's begin! What do you need built?"
        )
        conversation_state["context"] = f"AI: {initial_message}"
        return jsonify({"reply": initial_message, "phase": conversation_state["phase"]})

    context = conversation_state["context"]
    phase = conversation_state["phase"]
    awaiting = conversation_state["awaiting_confirmation"]

    # Handle "Approved" confirmations
    if user_input.lower() == "approved" and awaiting:
        conversation_state["awaiting_confirmation"] = False

        if phase == "Solution Idea":
            conversation_state["phase"] = "Feature Requirements"
            return jsonify({
                "reply": "✅ Great! Let’s talk about the business logic and start defining feature requirements.",
                "phase": conversation_state["phase"]
            })

        elif phase == "Feature Requirements":
            conversation_state["phase"] = "User Stories"
            # ✅ Don't return here — let LangChain generate user stories below
            user_input = "[System: Proceed to generate user stories]"  # or leave blank

        elif phase == "User Stories":
            conversation_state["phase"] = "Done"
            return jsonify({
                "reply": "✅ All done! Now, could you share your *name* and *phone number* so I can package this for the dev team?",
                "phase": conversation_state["phase"]
            })

        elif phase == "Done" and not conversation_state["user_details"].get("name"):
            return jsonify({
                "reply": "Please type your *name* and *phone number* like this:\n\nJohn Doe, +60123456789",
                "phase": phase
            })

    # Collect user details
    if phase == "Done" and not conversation_state["user_details"].get("name"):
        match = re.match(r"(.+),\s*(\+?\d{9,15})", user_input)
        if match:
            name, phone = match.groups()
            conversation_state["user_details"] = {"name": name.strip(), "phone": phone.strip()}
            conversation_state["email_ready"] = True
            return jsonify({
                "reply": f"✅ Thanks {name}!\n\nWould you like me to email this info to the Rangka Empat team now? (Type 'Approved')",
                "phase": phase
            })
        else:
            return jsonify({
                "reply": "❌ Format not recognized. Please type:\n\nJohn Doe, +60123456789",
                "phase": phase
            })

    # Handle final email approval
    if conversation_state["email_ready"] and user_input.lower() == "approved":
        body = f"""
Project Submission by AI Assistant

{conversation_state['context']}

User Name: {conversation_state['user_details']['name']}
Phone: {conversation_state['user_details']['phone']}
"""
        success = send_email(
            subject="New Project Proposal from REA Assistant",
            body=body,
            to_email="sivaprofess@gmail.com"
        )
        if success:
            return jsonify({
                "reply": "📧 Sent! Your project has been submitted to the team. They'll be in touch soon!",
                "phase": "Completed"
            })
        else:
            return jsonify({
                "reply": "⚠️ Failed to send the email. Please try again later or contact support.",
                "phase": phase
            })

    # 🚀 Detect if user input already contains a solution
    if phase == "Business Problem" and is_solution_statement(user_input):
        conversation_state["phase"] = "Feature Requirements"
        conversation_state["skipped_solution"] = True
        ai_reply = (
            "Thanks! It looks like you already have a solution in mind. ✅\n\n"
            "Let me ask a few business logic questions so I can define the feature requirements clearly."
        )
        context += f"\nUser: {user_input}\nAI: {ai_reply}"
        conversation_state["context"] = context
        return jsonify({"reply": ai_reply, "phase": "Feature Requirements"})

    # 🔗 Invoke Langchain
    result = chain.invoke({
        "context": context,
        "question": user_input,
        "phase": phase
    })

    ai_reply = getattr(result, "text", str(result))
    context += f"\nUser: {user_input}\nAI: {ai_reply}"
    conversation_state["context"] = context

    # Logic to set awaiting confirmations
    if phase == "Business Problem" and "solution" in ai_reply.lower():
        conversation_state["phase"] = "Solution Idea"
        conversation_state["awaiting_confirmation"] = True
        ai_reply += "\n\n👉 Type 'Approved' if this matches your idea."

    elif phase == "Feature Requirements":
        conversation_state["awaiting_confirmation"] = True
        ai_reply += "\n\n👉 Type 'Approved' to continue with user stories."

    elif phase == "User Stories":
        conversation_state["awaiting_confirmation"] = True
        ai_reply += "\n\n👉 Type 'Approved' if these user stories are accurate."

    return jsonify({"reply": ai_reply, "phase": conversation_state["phase"]})


if __name__ == "__main__":
    app.run(debug=True)
