from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import database as db
import json
from app.utils.helpers import load_csv, risk_label, generate_suggestions, explain_prediction
from app.services.prediction_service import PredictionService
from app.services.ai_service import AIService

student_blueprint = Blueprint("student_routes", __name__)

def login_required(role=None):
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("auth_routes.login_page"))
            if role and session.get("role") != role:
                flash("Access denied.", "danger")
                return redirect(url_for("auth_routes.login_page"))
            return f(*args, **kwargs)
        return wrapper
    return decorator

@student_blueprint.route("/student")
@login_required("student")
def student_dashboard():
    student = db.get_student_by_user_id(session["user_id"])
    row = {}
    score = None
    result = "Unknown"
    suggestions = []
    sensitivity = None
    remarks = []

    if student:
        student_id = student.get("student_code")
        df = load_csv()
        df["Student_ID"] = df["Student_ID"].astype(str).str.strip().str.upper()
        match = df[df["Student_ID"] == str(student_id).strip().upper()]
        
        if not match.empty:
            row = match.iloc[0].to_dict()
            score = PredictionService.predict_score(row)
            result = risk_label(score)
            suggestions = generate_suggestions(score if score else 0, row)
            remarks = db.get_remarks_by_student(student_id)
            sensitivity = PredictionService.calculate_sensitivity(row, score)

    return render_template("student_dashboard.html",
        student=student, row=row, score=score,
        result=result, suggestions=suggestions,
        sensitivity=sensitivity, remarks=remarks
    )

@student_blueprint.route("/parent")
@login_required("parent")
def parent_dashboard():
    parent = db.get_parent_by_user_id(session["user_id"])
    student_id = parent["student_code"] if parent else None
    
    row = {}
    score = None
    risk = "Unknown"
    suggestions = []
    feedbacks = []
    remarks = []
    
    if student_id:
        df = load_csv()
        df["Student_ID"] = df["Student_ID"].astype(str).str.strip().str.upper()
        match = df[df["Student_ID"] == str(student_id).strip().upper()]
        
        if not match.empty:
            row = match.iloc[0].to_dict()
            score = PredictionService.predict_score(row)
            risk = risk_label(score)
            suggestions = generate_suggestions(score if score else 0, row)
            remarks = db.get_remarks_by_student(student_id)
            
            show_recycle = request.args.get("show_recycle")
            feedbacks = db.get_feedbacks_by_parent(parent["id"], exclude_deleted=(show_recycle != "1"))

    return render_template("parent_dashboard.html",
        parent=parent, row=row, score=score, risk=risk,
        suggestions=suggestions, feedbacks=feedbacks, remarks=remarks
    )

@student_blueprint.route("/parent/mark-read", methods=["POST"])
@login_required("parent")
def parent_mark_read():
    parent = db.get_parent_by_user_id(session["user_id"])
    if parent:
        db.mark_feedback_as_read(parent["id"])
    return jsonify({"status": "success"})

@student_blueprint.route("/parent/log-hours", methods=["POST"])
@login_required("parent")
def parent_log_hours():
    parent = db.get_parent_by_user_id(session["user_id"])
    if parent:
        hours = float(request.form.get("hours", 0))
        db.log_study_hours(parent["student_code"], hours, "Parent")
        flash(f"Successfully logged {hours} hours of study.", "success")
    return redirect(url_for("student_routes.parent_dashboard"))

@student_blueprint.route("/parent/feedback", methods=["POST"])
@login_required("parent")
def parent_feedback():
    parent = db.get_parent_by_user_id(session["user_id"])
    if parent:
        msg = request.form.get("message", "").strip()
        sentiment = AIService.analyze_sentiment(msg)
        db.submit_feedback(parent["id"], parent["student_code"], msg, sentiment)
        flash("Message submitted to teacher. (Sentiment analyzed as " + sentiment + ")", "success")
    return redirect(url_for("student_routes.parent_dashboard"))

@student_blueprint.route("/career-guidance")
@login_required()
def career_guidance():
    role = session.get("role")
    student_id = None
    if role == "student":
        student = db.get_student_by_user_id(session["user_id"])
        student_id = student["student_code"] if student else None
    elif role == "parent":
        parent = db.get_parent_by_user_id(session["user_id"])
        student_id = parent["student_code"] if parent else None
        
    latest_career = db.get_latest_career_suggestion(student_id) if student_id else None
    suggestions = json.loads(latest_career["suggestion_json"]) if latest_career else None
    
    return render_template("career_guidance.html", suggestions=suggestions, latest_career=latest_career)

@student_blueprint.route("/api/career-suggest", methods=["POST"])
@login_required()
def api_career_suggest():
    role = session.get("role")
    student_id = None
    if role == "student":
        student = db.get_student_by_user_id(session["user_id"])
        student_id = student["student_code"] if student else None
    elif role == "parent":
        parent = db.get_parent_by_user_id(session["user_id"])
        student_id = parent["student_code"] if parent else None

    if not student_id:
        return jsonify({"error": "No Student Profile active"}), 400

    q_data = request.json
    # Build Gemini Career Prompt
    prompt = f"""
    Suggest 3 optimal career paths based on this academic profile:
    Interest: {q_data.get('interests', '')}
    Strengths: {q_data.get('strengths', '')}
    Extracurricular: {q_data.get('extra', '')}
    Study preference: {q_data.get('study_preference', '')}
    
    Return the response as a JSON array where each element is an object with:
    'title' (the career title),
    'description' (why it matches them),
    'skills' (array of skills to develop),
    'steps' (array of next steps to take).
    Only return the JSON list, no surrounding markdown, no backticks, no code block identifiers.
    """
    try:
        reply = AIService.call_gemini(prompt)
        # Parse output safely
        clean_json = re.sub(r"```[a-zA-Z]*", "", reply).strip()
        suggestions_list = json.loads(clean_json)
        db.save_career_suggestion(student_id, json.dumps(q_data), clean_json)
        return jsonify({"suggestions": suggestions_list})
    except Exception as e:
        # High quality fallback career suggestion
        fallback_suggestions = [
            {
                "title": f"Data Analyst / Scientist (Interests: {q_data.get('interests', 'General')})",
                "description": "Utilize analytical skills to parse business details and build models.",
                "skills": ["Python Coding", "SQL", "Statistics", "Machine Learning"],
                "steps": ["Complete introductory Python tutorials", "Analyze open-source dataset", "Engage in coding tournaments"]
            },
            {
                "title": "Software Engineer",
                "description": "Design premium, highly functional, and state-of-the-art web systems.",
                "skills": ["Algorithms", "Web Development", "Databases", "Problem Solving"],
                "steps": ["Create a custom web portfolio", "Contribute to GitHub", "Study data structures"]
            }
        ]
        db.save_career_suggestion(student_id, json.dumps(q_data), json.dumps(fallback_suggestions))
        return jsonify({"suggestions": fallback_suggestions})
import re
