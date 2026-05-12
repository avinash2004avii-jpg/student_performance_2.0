import re
from flask import Blueprint, request, jsonify, session
from app.services.ai_service import AIService
from app.services.prediction_service import PredictionService
from app.utils.helpers import load_csv, risk_label, generate_suggestions
import database as db

chatbot_blueprint = Blueprint("chatbot_routes", __name__)

@chatbot_blueprint.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_message = request.json.get("message", "").strip()
        user_message_lower = user_message.lower()

        print(f"\n[CHATBOT REQUEST] Message: '{user_message}'")

        if "user_id" not in session:
            return jsonify({"reply": "🔒 Please login first."})

        role = session.get("role")
        print(f"[CHATBOT INFO] User ID: {session['user_id']} | Role: {role}")



        # Load CSV for context
        try:
            df = load_csv()
        except Exception as e:
            print(f"[CHATBOT ERROR] Failed to load CSV: {e}")
            return jsonify({"reply": "⚠️ Error loading system database file. Please try again later."})

        # Context collection
        student_id = None
        student_context = None

        if role == "student" or role == "parent":
            user_record = db.get_student_by_user_id(session["user_id"]) if role == "student" else db.get_parent_by_user_id(session["user_id"])
            if user_record:
                student_id = str(user_record.get("student_code", "")).strip().upper()
        
        # Parse student ID mentioned in message (useful for teachers)
        if not student_id:
            match_id = re.search(r"student\s*(\d+)", user_message_lower)
            if match_id:
                student_id = match_id.group(1).upper()

        if student_id:
            df["Student_ID"] = df["Student_ID"].astype(str).str.strip().str.upper()
            match = df[df["Student_ID"] == student_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                score = PredictionService.predict_score(row)
                risk = risk_label(score)
                suggestions = generate_suggestions(score if score else 0, row)
                
                student_context = row.copy()
                student_context["score"] = score
                student_context["risk"] = risk
                student_context["suggestions"] = suggestions
                print(f"[CHATBOT INFO] Found context for Student ID: {student_id}")

        # Conversation History
        if "chat_history" not in session:
            session["chat_history"] = []
        
        history = session["chat_history"][-6:] # Last 3 turns
        
        # Call AI Service
        reply = AIService.get_chatbot_reply(
            user_message=user_message,
            session_history=history,
            user_role=role,
            student_context=student_context
        )

        # Generate dynamic suggestions
        user_message_lower = user_message.lower()
        if role == "teacher":
            if student_id:
                suggested_qs = [
                    f"Why is Student {student_id} at risk?",
                    f"Show improvement tips for Student {student_id}",
                    "Which class performs best?"
                ]
            elif "at risk" in user_message_lower or "at-risk" in user_message_lower:
                suggested_qs = [
                    "Why is attendance low?",
                    "Which class performs best?",
                    "Explain analytics"
                ]
            elif "attendance" in user_message_lower:
                suggested_qs = [
                    "Show at-risk students",
                    "Explain analytics",
                    "Which class performs best?"
                ]
            elif "analytics" in user_message_lower or "class average" in user_message_lower:
                suggested_qs = [
                    "Show at-risk students",
                    "Why is attendance low?",
                    "Which class performs best?"
                ]
            else:
                suggested_qs = [
                    "Show at-risk students",
                    "Which class performs best?",
                    "Explain analytics"
                ]
        else: # student or parent
            if "score" in user_message_lower or "predicted" in user_message_lower or "result" in user_message_lower or "marks" in user_message_lower:
                suggested_qs = [
                    "How can I improve?",
                    "Why am I at risk?",
                    "What is my attendance?"
                ]
            elif "improve" in user_message_lower or "tips" in user_message_lower or "suggest" in user_message_lower:
                suggested_qs = [
                    "What is my attendance?",
                    "What are my test scores?",
                    "Show study tips"
                ]
            else:
                suggested_qs = [
                    "What is my predicted score?",
                    "How can I improve?",
                    "What is my attendance?"
                ]

        # Save history
        session["chat_history"].append({"role": "user", "content": user_message})
        session["chat_history"].append({"role": "bot", "content": reply})
        session.modified = True

        return jsonify({"reply": reply, "suggestions": suggested_qs})

    except Exception as e:
        print("Chatbot Error:", e)
        return jsonify({"reply": "❌ Something went wrong with the AI assistant. Please try again."})
