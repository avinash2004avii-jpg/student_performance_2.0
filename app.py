"""
app.py  —  Student Performance Prediction System
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file)
import pandas as pd
import numpy as np
import joblib, os, io
import database as db
from flask import session
from flask import jsonify
import traceback
from dotenv import load_dotenv
import os
from google import genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import mailer

load_dotenv()  # ✅ MUST come first

# ✅ Create Gemini client (LATEST way)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = "sps_secret_key_change_in_production"

  # ✅ ADD THIS BEFORE using getenv()


# ── Paths ──────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "data",   "students_data.csv")
BULK_OUT  = os.path.join(BASE, "data",   "bulk_results.csv")
MDL_DIR   = os.path.join(BASE, "models")


# ── Load model ────────────────────────────────────────────────────
def load_model():
    p = os.path.join(MDL_DIR, "student_model.pkl")
    if not os.path.exists(p):
        return None, None
    return (joblib.load(os.path.join(MDL_DIR, "student_model.pkl")),
            joblib.load(os.path.join(MDL_DIR, "model_columns.pkl")))

model, model_columns = load_model()

db.create_tables()

def login_required(role=None):
    """Decorator factory for route protection."""
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login_page"))
            if role and session.get("role") != role:
                flash("Access denied.", "danger")
                return redirect(url_for("login_page"))
            return f(*args, **kwargs)
        return wrapper
    return decorator

@app.before_request
def auto_purge():
    # Run for both teacher and parent routes
    if request.path.startswith("/teacher") or request.path.startswith("/parent"):
        db.purge_deleted_feedback()

@app.route("/parent/delete-feedback/<int:fid>", methods=["POST"])
@login_required("parent")
def parent_delete_feedback(fid):
    db.soft_delete_feedback(fid)
    flash("Message moved to Recycle Bin.", "warning")
    return redirect(url_for("parent_dashboard", show_recycle="1"))

@app.route("/parent/restore-feedback/<int:fid>", methods=["POST"])
@login_required("parent")
def parent_restore_feedback(fid):
    db.restore_feedback(fid)
    flash("Message restored.", "success")
    return redirect(url_for("parent_dashboard"))

@app.context_processor
def inject_feedback_count():
    counts = {"open_feedback_count": 0, "unread_reply_count": 0}
    if "user_id" in session:
        role = session.get("role")
        if role == "teacher":
            counts["open_feedback_count"] = db.get_open_feedback_count()
        elif role == "parent":
            parent = db.get_parent_by_user_id(session["user_id"])
            if parent:
                counts["unread_reply_count"] = db.get_unread_reply_count(parent["id"])
    return counts


# ── Helpers ───────────────────────────────────────────────────────
def load_csv():
    df = pd.read_csv(DATA_FILE)
    df["Health_Issues"] = df["Health_Issues"].fillna("None")
    if "parent_email" in df.columns:
        df["parent_email"] = df["parent_email"].fillna("")
    return df

def fval(form, key, default=0):
    """Safe float from form — handles empty strings gracefully."""
    v = form.get(key, "").strip()
    try:
        return float(v) if v != "" else float(default)
    except (ValueError, TypeError):
        return float(default)

def ival(form, key, default=0):
    """Safe int from form — handles empty strings gracefully."""
    v = form.get(key, "").strip()
    try:
        return int(v) if v != "" else int(default)
    except (ValueError, TypeError):
        return int(default)

def risk_label(score):
    if score is None: return "Unknown"
    if score < 60:    return "At Risk"
    if score <= 75:   return "Average Performance"
    return "Safe"

def calculate_display_risk(df):
    """Vectorized calculation of display_score and risk for performance."""
    if df.empty:
        df["display_score"] = []
        df["risk"] = []
        df["risk_val"] = []
        return df
    
    # 1. Start with actual scores
    df["display_score"] = df["Final_Exam_Score"].astype(float)
    
    # 2. Identify which rows need prediction (actual score is 0)
    mask = df["display_score"] <= 0
    
    if mask.any() and model is not None:
        # 3. Batch feature engineering for subset
        sub = df[mask].copy()
        
        # Numeric extraction (handles both CSV and internal names)
        it1 = sub.get("internal_test 1", sub.get("internal1", pd.Series(0, index=sub.index))).astype(float)
        it2 = sub.get("internal_test 2", sub.get("internal2", pd.Series(0, index=sub.index))).astype(float)
        asgn = sub.get("Assignment_score", sub.get("assignment", pd.Series(0, index=sub.index))).astype(float)
        prev = sub.get("Previous_Exam_Score", sub.get("previous", pd.Series(0, index=sub.index))).astype(float)
        att = sub.get("Attendence", sub.get("attendance", pd.Series(75, index=sub.index))).astype(float)
        sh = sub.get("Study_hours", sub.get("study_hours", pd.Series(3, index=sub.index))).astype(float)
        slp = sub.get("Sleep_hours", sub.get("sleep_hours", pd.Series(7, index=sub.index))).astype(float)

        # Build feature set
        X_data = {
            "Study_hours": sh, "Sleep_hours": slp, "Attendence": att,
            "internal_test 1": it1, "internal_test 2": it2,
            "Assignment_score": asgn, "Previous_Exam_Score": prev,
            "total_internal": it1 + it2,
            "avg_internal": (it1 + it2) / 2,
            "study_x_attendance": sh * att / 100,
            "total_score": it1 + it2 + asgn + prev,
            "study_efficiency": sh / (slp + 1),
            "high_study": (sh > 4).astype(int),
        }
        
        # Categorical
        for col in ["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"]:
            X_data[col] = sub.get(col, pd.Series("None" if col=="Health_Issues" else "Yes", index=sub.index))
            
        X_df = pd.DataFrame(X_data)
        X_encoded = pd.get_dummies(X_df, columns=["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"], drop_first=True)
        
        # Align with model columns
        final_X = pd.DataFrame(0, index=sub.index, columns=model_columns)
        for col in model_columns:
            if col in X_encoded.columns:
                final_X[col] = X_encoded[col]
            elif col in X_df.columns:
                final_X[col] = X_df[col]
        
        # Batch Predict
        try:
            preds = model.predict(final_X)
            df.loc[mask, "display_score"] = np.round(preds, 1)
        except Exception as e:
            print(f"Batch Prediction Error: {e}")

    # 4. Vectorized Labels
    df["risk"] = np.where(df["display_score"] < 60, "At Risk",
                 np.where(df["display_score"] <= 75, "Average Performance", "Safe"))
    df["risk_val"] = np.where(df["display_score"] < 60, "1", "0")
    
    return df


def build_features(row):
    # Extract numerical inputs (handle both CSV and Form keys)
    it1  = float(row.get("internal_test 1",  row.get("internal1",  0)))
    it2  = float(row.get("internal_test 2",  row.get("internal2",  0)))
    asgn = float(row.get("Assignment_score", row.get("assignment", 0)))
    prev = float(row.get("Previous_Exam_Score", row.get("previous", 0)))
    att  = float(row.get("Attendence",  row.get("attendance",  75)))
    sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
    slp  = float(row.get("Sleep_hours", row.get("sleep_hours", 7)))

    # Required Features by user
    total_internal = it1 + it2
    avg_internal   = total_internal / 2

    # Build numeric dictionary
    data = {
        "Study_hours": sh, "Sleep_hours": slp, "Attendence": att,
        "internal_test 1": it1, "internal_test 2": it2,
        "Assignment_score": asgn, "Previous_Exam_Score": prev,
        "total_internal": total_internal,
        "avg_internal": avg_internal,
        "study_x_attendance": sh * att / 100,
        "total_score": it1 + it2 + asgn + prev,
        "study_efficiency": sh / (slp + 1),
        "high_study": 1 if sh > 4 else 0,
    }

    # Handle Categorical variables via manual dummy creation for consistency
    # This ensures we match the columns expected by the model
    hlth = row.get("Health_Issues", row.get("health", "None"))
    gender = row.get("Gender", "Male")
    internet = row.get("Internet_Access", "Yes")
    extra = row.get("Extracurricular_Activities", "No")

    # Add raw strings to data for pd.get_dummies
    data["Health_Issues"] = hlth
    data["Gender"] = gender
    data["Internet_Access"] = internet
    data["Extracurricular_Activities"] = extra

    df_row = pd.DataFrame([data])
    cols_to_encode = ["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"]
    # Match training script's drop_first=True
    df_encoded = pd.get_dummies(df_row, columns=cols_to_encode, drop_first=True)

    # ALIGNMENT: Ensure all columns from training exist and are in the correct order
    # Any missing dummy column is set to 0
    final_features = pd.DataFrame(0, index=[0], columns=model_columns)
    for col in model_columns:
        if col in df_encoded.columns:
            final_features[col] = df_encoded[col].values
        elif col in data:
            final_features[col] = data[col]

    return final_features

def predict_score(row):
    if model is None: return None
    return round(float(model.predict(build_features(row))[0]), 1)

def get_student_by_id(student_id):
    """Fetch student dictionary from CSV based on Student_ID."""
    try:
        df = load_csv()
        # Student_ID in CSV is numeric (or string depending on pandas)
        match = df[df["Student_ID"].astype(str) == str(student_id)]
        if match.empty:
            return None
        return match.iloc[0].to_dict()
    except Exception:
        return None

def generate_suggestions(score, row, class_avg=None):
    """Return list of personalised improvement tips based on student data."""
    att  = float(row.get("Attendence", row.get("attendance", 100)))
    sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
    it1  = float(row.get("internal_test 1", row.get("internal1", 0)))
    it2  = float(row.get("internal_test 2", row.get("internal2", 0)))
    asgn = float(row.get("Assignment_score", row.get("assignment", 0)))
    slp  = float(row.get("Sleep_hours", row.get("sleep_hours", 7)))

    # 1. Handle "At Risk" students with specific static-style tips as requested
    if score < 60:
        return [
            ("Attendance", f"Your attendance is quite low right now ({att}%). This can seriously affect your understanding of subjects. Try to attend classes regularly — even improving to 75–85% can make a big difference in your performance."),
            ("Academics (Low Marks)", "Your current scores show that some concepts may not be clear. Start revising basic topics daily and focus on understanding rather than memorizing. Even 1–2 hours of focused study can improve your marks significantly."),
            ("Study Hours", f"Studying {sh} hours is a good start, but increasing it slightly with proper focus can boost your performance. Try creating a simple daily study plan and stick to it."),
            ("Motivation / Overall", "Right now you are in the ‘At Risk’ category, but this is not permanent. With small consistent efforts, you can improve your performance step by step. Start with one subject at a time and build confidence.")
        ]

    # 2. Dynamic tips for Average/Safe students
    tips = []
    if att < 75:
        tips.append(("Attendance", f"Attendance is {att}% — below the 75% minimum. Missing class directly correlates with lower scores. Aim for at least 85% attendance."))
    
    if class_avg and score < class_avg:
        diff = round(class_avg - score, 1)
        tips.append(("Benchmark", f"Your predicted score is {diff} marks below the class average ({class_avg}). Try to identify subjects where you are losing marks and seek extra guidance."))

    if sh < 2:
        tips.append(("Study Time", f"Only {sh} hours of study per day is very low. Increasing to at least 3–4 hours daily can significantly improve performance."))
    elif sh < 3:
        tips.append(("Study Time", f"{sh} hours of study is below average. Try adding one focused study session per day."))
    
    if it2 < it1:
        tips.append(("Declining Trend", f"Internal Test 2 ({it2}) is lower than Internal Test 1 ({it1}). Performance is declining — review test 2 topics thoroughly and seek teacher help."))
    
    if asgn < 50:
        tips.append(("Assignments", f"Assignment score is only {asgn}/100. Completing assignments consistently is one of the easiest ways to improve your grade."))
    
    if slp < 6:
        tips.append(("Sleep", f"Only {slp} hours of sleep affects memory and focus. 7–8 hours of sleep helps retain what you study."))
    
    if it1 < 50 and it2 < 50:
        tips.append(("Core Concepts", "Both internal tests are below 50. Focus on understanding fundamentals rather than memorising — consider extra tutoring."))
    
    if not tips:
        tips.append(("Achievement", "Keep up the great work and maintain consistency!"))

    return tips

import json

def generate_advanced_insights(row):
    """Generate career, course, and behavioral analysis using Gemini."""
    try:
        # Prepare a clean data summary for the prompt
        data_summary = {
            "Internal 1": row.get("internal_test 1", row.get("internal1", 0)),
            "Internal 2": row.get("internal_test 2", row.get("internal2", 0)),
            "Assignment": row.get("Assignment_score", row.get("assignment", 0)),
            "Attendance": row.get("Attendence", row.get("attendance", 0)),
            "Study Hours": row.get("Study_hours", row.get("study_hours", 0)),
            "Sleep Hours": row.get("Sleep_hours", row.get("sleep_hours", 0)),
            "Previous Exam": row.get("Previous_Exam_Score", row.get("previous", 0)),
            "Health": row.get("Health_Issues", row.get("health", "None"))
        }

        prompt = f"""
        Analyze this student's academic performance and habits to provide specific guidance.
        Data: {json.dumps(data_summary)}

        Provide the following in a strictly structured JSON format:
        1. "careers": A list of 3 specific career paths suited to their strengths.
        2. "courses": A list of 3 recommended subjects or online course topics to improve or excel.
        3. "behavior": A short paragraph (2-3 sentences) analyzing their study habits and attendance patterns.

        JSON format example:
        {{
            "careers": ["Data Scientist", "Software Engineer", "Research Analyst"],
            "courses": ["Advanced Mathematics", "Python Programming", "Time Management"],
            "behavior": "Your high attendance shows commitment, but low study hours compared to sleep suggest you could optimize your schedule for better retention."
        }}
        Return ONLY the JSON.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Clean the response in case Gemini adds markdown code blocks
        clean_text = response.text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]
        
        return json.loads(clean_text.strip())
    except Exception as e:
        print(f"Error generating insights: {e}")
        return {
            "careers": ["Education", "General Management", "Public Service"],
            "courses": ["Foundational Mathematics", "Communication Skills", "Digital Literacy"],
            "behavior": "Unable to perform behavioral analysis at this time. Focus on maintaining a balanced study-life schedule."
        }
def analyze_sentiment(message):
    """Analyze sentiment of parent message using Gemini."""
    try:
        prompt = f"Analyze the sentiment of this parent message to a teacher: '{message}'. Categorize it as exactly one of: 'Urgent', 'Frustrated', 'Neutral', or 'Positive'. Return only the word."
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        sentiment = response.text.strip().replace("'", "").replace('"', "")
        if sentiment in ['Urgent', 'Frustrated', 'Neutral', 'Positive']:
            return sentiment
        return "Neutral"
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
        return "Neutral"

def calculate_sensitivity(row, current_score):
    """Calculate what's needed to reach 'Safe' (75) or 'Average' (60) status."""
    if current_score >= 75:
        return None # Already safe
    
    target = 75 if current_score < 75 else 80 # Aim for safe
    results = []
    
    # 1. Try Attendance
    test_row = row.copy()
    orig_att = float(row.get('Attendence', 75))
    if orig_att < 95:
        needed_att = min(100.0, orig_att + 10.0)
        test_row['Attendence'] = needed_att
        new_score = predict_score(test_row)
        if new_score and new_score > current_score:
            results.append({"metric": "Attendance", "from": orig_att, "to": needed_att, "gain": round(new_score - current_score, 1)})

    # 2. Try Study Hours
    test_row = row.copy()
    orig_sh = float(row.get('Study_hours', 3))
    if orig_sh < 8:
        needed_sh = orig_sh + 2.0
        test_row['Study_hours'] = needed_sh
        new_score = predict_score(test_row)
        if new_score and new_score > current_score:
            results.append({"metric": "Study Hours", "from": orig_sh, "to": needed_sh, "gain": round(new_score - current_score, 1)})
            
    # 3. Try Next Internal (simulating IT2 or Assignment improvement)
    test_row = row.copy()
    orig_it2 = float(row.get('internal_test 2', 0))
    if orig_it2 < 90:
        needed_it2 = min(100.0, orig_it2 + 15.0)
        test_row['internal_test 2'] = needed_it2
        new_score = predict_score(test_row)
        if new_score and new_score > current_score:
            results.append({"metric": "Next Internal Test", "from": orig_it2, "to": needed_it2, "gain": round(new_score - current_score, 1)})

    return results



# ════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    if "user_id" in session:
        role = session.get("role")
        if role == "admin":   return redirect(url_for("admin_dashboard"))
        if role == "teacher": return redirect(url_for("teacher_dashboard"))
        if role == "student": return redirect(url_for("student_dashboard"))
        if role == "parent":  return redirect(url_for("parent_dashboard"))
    return render_template("home.html")


# ════════════════════════════════════════════════════════════════════
# AUTH — Login / Signup / Logout
# ════════════════════════════════════════════════════════════════════
@app.route("/login", methods=["GET", "POST"])
def login_page():
    """Generic login — redirects to role-specific page."""
    return render_template("login.html")

@app.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "admin":
            if user["status"] != "active":
                flash(f"Admin account is {user['status']}. Please check database.", "danger")
                return redirect(url_for("login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("admin_dashboard"))
        flash("Invalid credentials or not an admin account.", "danger")
    return render_template("login_admin.html")

@app.route("/login/teacher", methods=["GET", "POST"])
def login_teacher():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "teacher":
            if user["status"] == "pending":
                flash("Your account is pending admin approval.", "warning")
                return redirect(url_for("login_page"))
            if user["status"] == "deactivated":
                flash("Your account has been deactivated.", "danger")
                return redirect(url_for("login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("teacher_dashboard"))
        flash("Invalid credentials or not a teacher account.", "danger")
    return render_template("login_teacher.html")

@app.route("/login/student", methods=["GET", "POST"])
def login_student():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "student":
            if user["status"] == "deactivated":
                flash("Your account has been deactivated.", "danger")
                return redirect(url_for("login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("student_dashboard"))
        flash("Invalid credentials or not a student account.", "danger")
    return render_template("login_student.html")

@app.route("/login/parent", methods=["GET", "POST"])
def login_parent():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "parent":
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("parent_dashboard"))
        flash("Invalid credentials or not a parent account.", "danger")
    return render_template("login_parent.html")


@app.route("/signup/teacher", methods=["GET", "POST"])
def signup_teacher():
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]
        confirm  = request.form["confirm"]
        name     = request.form["name"].strip()
        subject  = request.form.get("subject", "General").strip()

        if password != confirm:
            flash("Passwords do not match.", "danger")
        elif db.username_exists(username):
            flash("Username already taken.", "danger")
        elif db.email_exists(email):
            flash("Email already registered.", "danger")
        else:
            success, msg = db.signup_teacher(username, email, password, name, subject)
        if success:
            flash("Registration successful! Your account is pending admin approval.", "success")
            return redirect(url_for("login_page"))
        flash(msg, "danger")
    return render_template("signup_teacher.html")


@app.route("/signup/student", methods=["GET", "POST"])
def signup_student():
    teachers = db.get_all_teachers_simple()
    if request.method == "POST":
        username     = request.form["username"].strip()
        email        = request.form["email"].strip()
        password     = request.form["password"]
        confirm      = request.form["confirm"]
        name         = request.form["name"].strip()
        student_code = request.form["student_code"].strip()
        class_       = request.form.get("class_", "")
        section      = request.form.get("section", "")
        teacher_id   = request.form.get("teacher_id") or None

        if password != confirm:
            flash("Passwords do not match.", "danger")
        elif db.username_exists(username):
            flash("Username already taken.", "danger")
        elif db.email_exists(email):
            flash("Email already registered.", "danger")
        else:
            # ✅ Check if Student ID exists in the master school database (CSV)
            df = load_csv()
            if student_code not in df["Student_ID"].astype(str).values:
                flash(f"Student ID {student_code} was not found in our records. Please ensure it matches exactly with school records.", "danger")
            else:
                ok, msg = db.signup_student(username, email, password, name,
                                            student_code, class_, section, teacher_id)
                if ok:
                    flash("Account created! Please log in.", "success")
                    return redirect(url_for("login_student"))
                
                # Give a clear message for the most common failure
                if "student_code" in msg or "UNIQUE" in msg:
                    flash("That Student ID is already registered. Check your roll number.", "danger")
                else:
                    flash(msg, "danger")

    return render_template("signup_student.html", teachers=teachers)


@app.route("/signup/parent", methods=["GET", "POST"])
def signup_parent():
    if request.method == "POST":
        username     = request.form["username"].strip()
        email        = request.form["email"].strip()
        password     = request.form["password"]
        confirm      = request.form["confirm"]
        parent_name  = request.form["parent_name"].strip()
        student_code = request.form["student_code"].strip()

        if password != confirm:
            flash("Passwords do not match.", "danger")
        elif db.username_exists(username):
            flash("Username already taken.", "danger")
        elif db.email_exists(email):
            flash("Email already registered.", "danger")
        else:
            df = load_csv()
            if student_code not in df["Student_ID"].astype(str).values:
                flash(f"Student ID {student_code} not found.", "danger")
            else:
                ok, msg = db.signup_parent(username, email, password, parent_name, student_code)
                if ok:
                    flash("Account created! Please log in.", "success")
                    return redirect(url_for("login_parent"))
                flash(msg, "danger")

    return render_template("signup_parent.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ════════════════════════════════════════════════════════════════════
# ADMIN
# ════════════════════════════════════════════════════════════════════
@app.route("/admin")
@login_required("admin")
def admin_dashboard():
    # 1. User stats
    all_users = db.get_all_users()
    pending_count = len([u for u in all_users if u["status"] == "pending"])
    total_teachers = len([u for u in all_users if u["role"] == "teacher"])
    
    # 2. Student performance stats
    df = load_csv()
    total_students = len(df)
    
    # Risk calculation using project standard thresholds (<70 is Risk, >85 is Safe)
    risk_count = len(df[df["Final_Exam_Score"] < 70])
    safe_count = len(df[df["Final_Exam_Score"] >= 85])
    avg_score = round(df["Final_Exam_Score"].mean(), 1) if not df.empty else 0
    
    return render_template("admin_dashboard.html", 
                           pending_count=pending_count,
                           total_students=total_students,
                           total_teachers=total_teachers,
                           risk_count=risk_count,
                           safe_count=safe_count,
                           avg_score=avg_score)

@app.route("/admin/users")
@login_required("admin")
def admin_users():
    role_filter = request.args.get("role")
    status_filter = request.args.get("status")
    
    users = db.get_all_users()
    if role_filter:
        users = [u for u in users if u["role"] == role_filter]
    if status_filter:
        users = [u for u in users if u["status"] == status_filter]
        
    return render_template("admin_users.html", users=users)

@app.route("/admin/user-action/<int:user_id>", methods=["POST"])
@login_required("admin")
def admin_user_action(user_id):
    action = request.form.get("action")
    if action == "approve":
        db.update_user_status(user_id, "active")
        flash("User approved successfully.", "success")
    elif action == "deactivate":
        db.update_user_status(user_id, "deactivated")
        flash("User deactivated successfully.", "warning")
    elif action == "activate":
        db.update_user_status(user_id, "active")
        flash("User activated successfully.", "success")
    elif action == "delete":
        db.delete_user(user_id)
        flash("User deleted successfully.", "danger")
    elif action == "reset":
        new_pass = request.form.get("new_password")
        if new_pass:
            db.reset_user_password(user_id, new_pass)
            flash("Password reset successfully.", "success")
        else:
            flash("New password cannot be empty.", "warning")
    return redirect(url_for("admin_users"))
    df = load_csv()
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", "Safe")
    return render_template("admin_dashboard.html",
        total_students=len(df),
        risk_count=len(df[df["risk"] == "At Risk"]),
        total_teachers=len(db.get_all_teachers()),
        users=db.get_all_users(),
        teachers=db.get_all_teachers(),
    )

@app.route("/admin/delete-user/<int:uid>")
@login_required("admin")
def admin_delete_user(uid):
    if uid == session["user_id"]:
        flash("You cannot delete your own account.", "danger")
    else:
        db.delete_user(uid)
        flash("User deleted.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/students")
@login_required("admin")
def admin_students():
    q = request.args.get("q", "")
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    sel_filter = request.args.get("filter", "")
    
    df = load_csv()
    df = calculate_display_risk(df)
    
    # Get unique values for filters
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    if q:
        df = df[df["Student_ID"].astype(str).str.contains(q, na=False)]
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
    if sel_filter == "risk":
        df = df[df["display_score"] < 70]

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template("students_rows.html", students=df.to_dict(orient="records"))

    return render_template("students_table.html",
                           students=df.to_dict(orient="records"), 
                           q=q, classes=classes, sections=sections,
                           sel_class=sel_class, sel_section=sel_section)

@app.route("/admin/add-student", methods=["GET", "POST"])
@login_required("admin")
def admin_add_student():
    if request.method == "POST":
        df = load_csv()
        new = {
            "Student_ID": request.form["student_id"],
            "Class": request.form["class_"], "section": request.form["section"],
            "Age": int(request.form.get("age", 14)),
            "Gender": request.form.get("gender", "Male"),
            "Study_hours": float(request.form.get("study_hours", 3)),
            "Sleep_hours": float(request.form.get("sleep_hours", 7)),
            "Parent_Education_Level": request.form.get("parent_edu", "High School"),
            "Health_Issues": request.form.get("health", "None"),
            "Internet_Access": request.form.get("internet", "Yes"),
            "Attendence": float(request.form.get("attendance", 75)),
            "internal_test 1": float(request.form.get("internal1", 0)),
            "internal_test 2": float(request.form.get("internal2", 0)),
            "Assignment_score": float(request.form.get("assignment", 0)),
            "Extracurricular_Activities": request.form.get("extra", "No"),
            "Previous_Exam_Score": float(request.form.get("previous", 0)),
            "Final_Exam_Score": float(request.form.get("final_score", 0)),
            "parent_email": request.form.get("parent_email", "").strip()
        }
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df = df.drop_duplicates(subset=["Student_ID"], keep="last")
        df.to_csv(DATA_FILE, index=False)
        flash(f"Student {new['Student_ID']} added.", "success")
        return redirect(url_for("admin_students"))
    return render_template("add_student.html")

@app.route("/admin/delete-student/<sid>")
@login_required("admin")
def admin_delete_student(sid):
    df = load_csv()
    df = df[df["Student_ID"].astype(str) != sid]
    df.to_csv(DATA_FILE, index=False)
    flash("Student removed.", "success")
    return redirect(url_for("admin_students"))

@app.route("/admin/upload-students", methods=["POST"])
@login_required("admin")
def admin_upload_students():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("admin_students"))
    new_df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
    new_df["Health_Issues"] = new_df["Health_Issues"].fillna("None")
    if "parent_email" not in new_df.columns:
        new_df["parent_email"] = ""
    else:
        new_df["parent_email"] = new_df["parent_email"].fillna("")
    
    existing = load_csv()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Student_ID"], keep="last")
    combined.to_csv(DATA_FILE, index=False)
    flash(f"Uploaded {len(new_df)} rows.", "success")
    return redirect(url_for("admin_students"))


# ════════════════════════════════════════════════════════════════════
# TEACHER
# ════════════════════════════════════════════════════════════════════
@app.route("/teacher")
@login_required("teacher")
def teacher_dashboard():
    teacher = db.get_teacher_by_user_id(session["user_id"])
    df = load_csv()
    
    # Get unique values for filters
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    # Apply filters from request
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
        
    df = calculate_display_risk(df)
    
    total = len(df)
    at_risk_count = len(df[df["display_score"] < 70])
    avg_score = round(df["display_score"].mean(), 1) if not df.empty else 0
    avg_att = round(df["Attendence"].mean(), 1) if not df.empty else 0
    risk_pct = round((at_risk_count / total * 100), 1) if total > 0 else 0
    
    # Limit to 10 for dashboard preview
    risk_students = df[df["risk"] == "At Risk"].head(10).to_dict(orient="records")
    
    # Get Parent Feedback
    feedback_list = db.get_all_feedback(status="open")
    
    return render_template("teacher_dashboard.html",
        teacher=teacher,
        total=total,
        at_risk=at_risk_count,
        avg_score=avg_score, 
        avg_att=avg_att, 
        risk_pct=risk_pct,
        risk_students=risk_students,
        classes=classes,
        sections=sections,
        sel_class=sel_class,
        sel_section=sel_section,
        feedback_list=feedback_list
    )

@app.route("/teacher/predict", methods=["GET", "POST"])
@login_required("teacher")
def teacher_predict():
    result = None
    score  = None
    suggestions = []
    if request.method == "POST":
        try:
            # Server-side validation for negative numbers
            for field in ['internal1', 'internal2', 'assignment', 'previous', 'attendance', 'study_hours', 'sleep_hours']:
                val = fval(request.form, field)
                if val < 0:
                    flash(f"Invalid input: {field.replace('_', ' ').title()} cannot be negative.", "danger")
                    return render_template("predict_single.html", result=None, score=None, suggestions=[], f=request.form)

            score  = predict_score(request.form)
            result = risk_label(score)
            if score is not None:
                df = load_csv()
                class_avg = round(float(df["Final_Exam_Score"].mean()), 1) if not df.empty else 0
                suggestions = generate_suggestions(score, request.form, class_avg=class_avg)
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
    return render_template("predict_single.html",
                           result=result, score=score, suggestions=suggestions,
                           f=request.form if request.method == 'POST' else {})

@app.route("/teacher/auto_predict", methods=["GET", "POST"])
@login_required("teacher")
def teacher_auto_predict():
    result = None
    score  = None
    suggestions = []
    student = None
    
    sid = (request.form.get("student_id") or request.args.get("student_id") or "").strip()
    
    if sid:
        student = get_student_by_id(sid)
        if student:
            try:
                # Use model directly for 2 decimal places as requested
                raw_pred = float(model.predict(build_features(student))[0])
                score = round(raw_pred, 2)
                
                # User requested thresholds for this feature: <60, 60-75, >75
                if score < 60:
                    result = "At Risk"
                elif score <= 75:
                    result = "Average Performance"
                else:
                    result = "Safe"
                
                df = load_csv()
                class_avg = round(float(df["Final_Exam_Score"].mean()), 1) if not df.empty else 0
                suggestions = generate_suggestions(score, student, class_avg=class_avg)
                if result == "At Risk":
                    parent_email = db.get_parent_email_by_student_code(sid)
                    if parent_email:
                        subject = f"Academic Alert: {sid} is At Risk 🎓"
                        # Generate a mock link for the roadmap (or use actual route if hosted)
                        roadmap_link = f"{request.url_root}teacher/intervention-plan/{sid}"
                        content = mailer.get_risk_alert_template(sid, score, roadmap_link)
                        mailer.send_email_async(parent_email, subject, content)
            except Exception as e:
                flash(f"Prediction error: {e}", "danger")
        else:
            flash(f"Student ID '{sid}' not found in our records.", "warning")
                
    return render_template("auto_predict.html",
                           result=result, score=score, 
                           suggestions=suggestions, student=student)

@app.route("/teacher/download-report")
@login_required("teacher")
def download_report():
    sid = request.args.get("student_id")
    # Also support getting data from manual prediction if provided in args
    it1 = fval(request.args, "internal1")
    it2 = fval(request.args, "internal2")
    asgn = fval(request.args, "assignment")
    prev = fval(request.args, "previous")
    att = fval(request.args, "attendance")
    sh = fval(request.args, "study_hours")
    slp = fval(request.args, "sleep_hours", 7)
    hlth = request.args.get("health", "None")
    
    # If student_id is provided, try to fetch missing data from DB
    student_data = None
    final_actual = 0
    if sid:
        student_data = get_student_by_id(sid)
        if student_data:
            # Override with DB data if not provided in args
            it1 = float(student_data.get("internal_test 1", it1))
            it2 = float(student_data.get("internal_test 2", it2))
            asgn = float(student_data.get("Assignment_score", asgn))
            prev = float(student_data.get("Previous_Exam_Score", prev))
            att = float(student_data.get("Attendence", att))
            sh = float(student_data.get("Study_hours", sh))
            slp = float(student_data.get("Sleep_hours", slp))
            hlth = student_data.get("Health_Issues", hlth)
            final_actual = float(student_data.get("Final_Exam_Score", 0))
    else:
        sid = "Manual Entry"

    # Predict
    input_row = {
        "internal1": it1, "internal2": it2, "assignment": asgn,
        "previous": prev, "attendance": att, "study_hours": sh,
        "sleep_hours": slp, "health": hlth
    }
    score = predict_score(input_row)
    result = risk_label(score)
    
    df = load_csv()
    class_avg = round(float(df["Final_Exam_Score"].mean()), 1) if not df.empty else 0
    suggestions = generate_suggestions(score, input_row, class_avg=class_avg)

    # Benchmark Analysis
    percentile = 0
    if not df.empty:
        percentile = round((df[df["Final_Exam_Score"] < score].shape[0] / df.shape[0]) * 100, 1)

    # Generate Graph
    plt.figure(figsize=(6, 4))
    bars = ['IT 1', 'IT 2', 'Assignment', 'Previous', 'Final']
    vals = [it1, it2, asgn, prev, final_actual if final_actual > 0 else score]
    colors_list = ['#7c3aed', '#a855f7', '#f59e0b', '#10b981', '#ef4444']
    
    plt.bar(bars, vals, color=colors_list)
    plt.title('Student Performance Graph', fontsize=14, fontweight='bold')
    plt.ylabel('Marks')
    plt.ylim(0, 110)
    
    for i, v in enumerate(vals):
        plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()
    img_buf.seek(0)

    # PDF Creation
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=24, spaceAfter=20)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=18, spaceBefore=15, spaceAfter=10)
    normal_style = styles['Normal']
    bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold')

    elements = []
    
    # Header
    elements.append(Paragraph("Student Performance Report", title_style))
    
    # Student Info
    info_data = [
        [f"Student ID: {sid}"],
        [f"Final Exam Score: {final_actual if final_actual > 0 else score}"],
        [f"Attendance: {att}%"],
        [f"Study Hours: {sh}"]
    ]
    for row in info_data:
        elements.append(Paragraph(row[0], bold_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Prediction
    elements.append(Paragraph("Prediction", section_style))
    elements.append(Paragraph(f"Predicted Score: {score}", normal_style))
    elements.append(Paragraph(f"Risk Level: {result}", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Benchmark
    elements.append(Paragraph("Benchmark Analysis", section_style))
    elements.append(Paragraph(f"Class Average: {class_avg}", normal_style))
    diff = round(score - class_avg, 1)
    status = "ABOVE" if diff >= 0 else "BELOW"
    elements.append(Paragraph(f"You are {status} average by {abs(diff)} marks", normal_style))
    elements.append(Paragraph(f"Better than {percentile}% of students", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Visual Analysis
    elements.append(Paragraph("Performance Visual Analysis", section_style))
    elements.append(Image(img_buf, width=5*inch, height=3.5*inch))
    elements.append(Spacer(1, 0.2 * inch))

    # Suggestions
    elements.append(Paragraph("Suggestions", section_style))
    for title, tip in suggestions:
        elements.append(Paragraph(f"• <b>{title}:</b> {tip}", normal_style))

    doc.build(elements)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"student_{sid}_report.pdf", mimetype='application/pdf')

@app.route("/teacher/bulk-predict", methods=["GET", "POST"])
@login_required("teacher")
def teacher_bulk_predict():
    if request.method == "GET":
        return render_template("bulk_predict.html")

    if model is None:
        flash("Model not found. Run train_model.py first.", "danger")
        return render_template("bulk_predict.html")

    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return render_template("bulk_predict.html")

    try:
        df = pd.read_csv(file) if file.filename.lower().endswith(".csv") else pd.read_excel(file)
        df["Health_Issues"] = df["Health_Issues"].fillna("None")
    except Exception as e:
        flash(f"Could not read file: {e}", "danger")
        return render_template("bulk_predict.html")

    results = []
    for _, row in df.iterrows():
        try:
            score = predict_score(row.to_dict())
        except Exception as e:
            print(f"Row Prediction Error: {e}")
            score = None
        risk = risk_label(score)
        results.append({
            "Student_ID":      row.get("Student_ID", "—"),
            "Predicted_Score": score if score is not None else "Error",
            "Risk":            risk,
            "Attendance":      row.get("Attendence", "—"),
            "Study_Hours":     row.get("Study_hours", "—"),
            "Internal_1":      row.get("internal_test 1", "—"),
            "Internal_2":      row.get("internal_test 2", "—"),
            "Assignment":      row.get("Assignment_score", "—"),
            "Previous":        row.get("Previous_Exam_Score", "—"),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(BULK_OUT, index=False)

    valid_scores = [r["Predicted_Score"] for r in results if isinstance(r["Predicted_Score"], float)]
    avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else "—"

    return render_template("bulk_predict.html",
        results=results, has_results=True,
        total=len(results),
        at_risk=sum(1 for r in results if r["Risk"] == "At Risk"),
        avg_score=avg_score,
    )

@app.route("/teacher/bulk-predict/download")
@login_required("teacher")
def bulk_download():
    if not os.path.exists(BULK_OUT):
        flash("No results to download yet.", "warning")
        return redirect(url_for("teacher_bulk_predict"))
    return send_file(BULK_OUT, as_attachment=True, download_name="predictions.csv")


@app.route("/teacher/students")
@login_required("teacher")
def teacher_students():
    q = request.args.get("q", "")
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    risk_filter = request.args.get("risk", "")
    
    df = load_csv()
    df = calculate_display_risk(df)
    
    # Get unique values for filters
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    if q:
        df = df[df["Student_ID"].astype(str) == str(q)]
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
    if risk_filter:
        df = df[df["risk_val"] == risk_filter]
        
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return render_template("students_rows.html", students=df.to_dict(orient="records"))

    return render_template("students_table.html",
                           students=df.to_dict(orient="records"), 
                           q=q, classes=classes, sections=sections,
                           sel_class=sel_class, sel_section=sel_section)

@app.route("/teacher/delete-student/<sid>")
@login_required("teacher")
def teacher_delete_student(sid):
    df = load_csv()
    df = df[df["Student_ID"].astype(str) != str(sid)]
    df.to_csv(DATA_FILE, index=False)
    flash(f"Student {sid} removed successfully.", "success")
    return redirect(url_for("teacher_students"))


# ════════════════════════════════════════════════════════════════════
# TEACHER — Add student(s)
# ════════════════════════════════════════════════════════════════════
@app.route("/teacher/add-student", methods=["GET", "POST"])
@login_required("teacher")
def teacher_add_student():
    if request.method == "POST":
        df = load_csv()
        f = request.form
        new = {
            "Student_ID":              f["student_id"].strip(),
            "Class":                   f.get("class_", "9th"),
            "section":                 f.get("section", "A"),
            "Age":                     ival(f, "age", 14),
            "Gender":                  f.get("gender", "Male"),
            "Study_hours":             fval(f, "study_hours", 3),
            "Sleep_hours":             fval(f, "sleep_hours", 7),
            "Parent_Education_Level":  f.get("parent_edu", "High School"),
            "Health_Issues":           f.get("health", "None"),
            "Internet_Access":         f.get("internet", "Yes"),
            "Attendence":              fval(f, "attendance", 75),
            "internal_test 1":         fval(f, "internal1", 0),
            "internal_test 2":         fval(f, "internal2", 0),
            "Assignment_score":        fval(f, "assignment", 0),
            "Extracurricular_Activities": f.get("extra", "No"),
            "Previous_Exam_Score":     fval(f, "previous", 0),
            "Final_Exam_Score":        fval(f, "final_score", 0),
            "parent_email":            f.get("parent_email", "").strip()
        }
        sid = new["Student_ID"]
        if not sid:
            flash("Student ID is required.", "danger")
            return render_template("teacher_add_student.html")
        if sid in df["Student_ID"].astype(str).values:
            flash(f"Student ID {sid} already exists.", "danger")
            return render_template("teacher_add_student.html")
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        flash(f"Student {sid} added successfully.", "success")
        return redirect(url_for("teacher_students"))
    return render_template("teacher_add_student.html")



@app.route("/teacher/upload_csv", methods=["POST"])
@login_required("teacher")
def teacher_upload_csv():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected for upload.", "danger")
        return redirect(url_for("teacher_add_student"))
    
    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are allowed.", "danger")
        return redirect(url_for("teacher_add_student"))

    try:
        df_new = pd.read_csv(file)
        
        col_map = {
            'student_id': 'Student_ID',
            'class': 'Class',
            'section': 'section',
            'attendance': 'Attendence',
            'internal_test_1': 'internal_test 1',
            'internal_test_2': 'internal_test 2',
            'assignment_score': 'Assignment_score',
            'previous_exam_score': 'Previous_Exam_Score',
            'study_hours': 'Study_hours',
            'sleep_hours': 'Sleep_hours',
            'health_issues': 'Health_Issues',
            'parent_email': 'parent_email'
        }
        
        # Check if required columns (keys in col_map) exist in uploaded CSV
        # We'll be flexible and allow case-insensitive matches
        df_new.columns = [c.lower().strip() for c in df_new.columns]
        missing = [c for c in col_map.keys() if c not in df_new.columns]
        
        if missing:
            flash(f"CSV is missing required columns: {', '.join(missing)}", "danger")
            return redirect(url_for("teacher_add_student"))

        # Rename to match our data schema
        df_new = df_new.rename(columns=col_map)
        
        # Select only relevant columns
        cols_to_keep = list(col_map.values())
        df_new = df_new[cols_to_keep]

        # Data Cleaning
        df_new['Health_Issues'] = df_new['Health_Issues'].fillna("None")
        df_new['Final_Exam_Score'] = 0 # Default if unknown
        
        # Numeric conversion and default 0
        numeric_cols = ['Attendence', 'internal_test 1', 'internal_test 2', 'Assignment_score', 
                        'Previous_Exam_Score', 'Study_hours', 'Sleep_hours', 'Final_Exam_Score']
        for col in numeric_cols:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0)

        # Merge with existing data
        df_existing = load_csv()
        
        # Create backup before modifying database
        import shutil
        shutil.copy(DATA_FILE, DATA_FILE.replace(".csv", "_backup.csv"))
        
        # Avoid duplicate student IDs - keep newest
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Student_ID"], keep="last")
        
        df_combined.to_csv(DATA_FILE, index=False)
        flash(f"Successfully processed {len(df_new)} students from CSV.", "success")
        
    except Exception as e:
        flash(f"Error processing CSV: {e}", "danger")
        
    return redirect(url_for("teacher_students"))

@app.route("/teacher/delete-all-students", methods=["POST"])
@login_required("teacher")
def teacher_delete_all_students():
    try:
        df = load_csv()
        if not df.empty:
            # Create backup before deleting
            import shutil
            backup_path = DATA_FILE.replace(".csv", "_backup.csv")
            shutil.copy(DATA_FILE, backup_path)
            
        df_empty = pd.DataFrame(columns=df.columns)
        df_empty.to_csv(DATA_FILE, index=False)
        flash("All student records have been cleared. <a href='/teacher/undo-delete' style='color: #ffaa00; font-weight: bold; text-decoration: underline;'>Undo?</a>", "success")
    except Exception as e:
        flash(f"Error clearing records: {e}", "danger")
    return redirect(url_for("teacher_add_student"))

@app.route("/teacher/undo-delete")
@login_required("teacher")
def teacher_undo_delete():
    try:
        import shutil
        backup_path = DATA_FILE.replace(".csv", "_backup.csv")
        if os.path.exists(backup_path):
            shutil.copy(backup_path, DATA_FILE)
            os.remove(backup_path) # Clean up after undo
            flash("Action undone! Your records have been restored.", "success")
        else:
            flash("No backup found to undo.", "warning")
    except Exception as e:
        flash(f"Error undoing action: {e}", "danger")
    return redirect(url_for("teacher_add_student"))

@app.route("/teacher/sample_csv")
@login_required("teacher")
def teacher_sample_csv():
    headers = "student_id,class,section,attendance,internal_test_1,internal_test_2,assignment_score,previous_exam_score,study_hours,sleep_hours,health_issues,parent_email\n"
    sample_row = "S9999,10th,A,85.5,70,75,80,72,4,7,None,parent@example.com"
    output = io.BytesIO()
    output.write((headers + sample_row).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="sample_students.csv")


@app.route("/teacher/bulk-predict-template")
@login_required("teacher")
def teacher_bulk_predict_template():
    headers = "Student_ID,Attendence,Study_hours,Sleep_hours,internal_test 1,internal_test 2,Assignment_score,Previous_Exam_Score,Gender,Health_Issues,Internet_Access,Extracurricular_Activities\n"
    sample_row = "S1001,85,4,7,75,80,90,72,Male,None,Yes,No"
    output = io.BytesIO()
    output.write((headers + sample_row).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="bulk_predict_template.csv")


# ════════════════════════════════════════════════════════════════════
# STUDENT
# ════════════════════════════════════════════════════════════════════
@app.route("/student")
@login_required("student")
def student_dashboard():
    student = db.get_student_by_user_id(session["user_id"])
    df = load_csv()

    row = None
    score = None
    suggestions = []
    result = None

    sensitivity = []
    if student:
        match = df[df["Student_ID"].astype(str) == str(student["student_code"])]
        if not match.empty:
            row   = match.iloc[0].to_dict()
            score = predict_score(row)
            result = risk_label(score)
            if score is not None:
                if score < 80:
                    suggestions = generate_suggestions(score, row)
                sensitivity = calculate_sensitivity(row, score)

    return render_template("student_dashboard.html",
        student=student, row=row, score=score,
        result=result, suggestions=suggestions,
        sensitivity=sensitivity
    )

def explain_prediction(row):
    reasons = []

    att = float(row.get("Attendence", 100))
    sh = float(row.get("Study_hours", 5))
    it1 = float(row.get("internal_test 1", 0))
    it2 = float(row.get("internal_test 2", 0))
    asgn = float(row.get("Assignment_score", 100))

    if att < 75:
        reasons.append(f"📅 Your attendance is {att}%, which is below 75%. This affects your understanding of subjects.")

    if sh < 3:
        reasons.append(f"📚 You study only {sh} hours/day. Increasing study time can improve performance.")

    if it2 < it1:
        reasons.append(f"📉 Your marks dropped from Internal 1 ({it1}) to Internal 2 ({it2}).")

    if asgn < 50:
        reasons.append(f"📝 Your assignment score is {asgn}, which is low.")

    if not reasons:
        reasons.append("✅ Your performance is stable and good.")

    return reasons

import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def create_performance_graph(student_id, row):
    """Generates a performance bar chart for the student."""
    components = ['IT 1', 'IT 2', 'Assignment', 'Previous', 'Final']
    marks = [
        float(row.get('internal_test 1', 0)),
        float(row.get('internal_test 2', 0)),
        float(row.get('Assignment_score', 0)),
        float(row.get('Previous_Exam_Score', 0)),
        float(row.get('Final_Exam_Score', 0))
    ]
    
    plt.figure(figsize=(6, 4))
    colors = ['#7b2cff', '#a64cff', '#f59e0b', '#10b981', '#ef4444']
    plt.bar(components, marks, color=colors)
    plt.xlabel('Academic Components')
    plt.ylabel('Marks')
    plt.title('Student Performance Graph')
    plt.ylim(0, 110)
    
    # Add values on top of bars
    for i, v in enumerate(marks):
        plt.text(i, v + 2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
        
    # Ensure temp directory exists
    temp_dir = os.path.join(BASE, "static", "temp_reports")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    img_path = os.path.join(temp_dir, f"performance_{student_id}.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=100)
    plt.close()
    return img_path

def build_student_report_pdf(student_id, row, predicted_score, risk,
                             suggestions, class_avg,
                             benchmark_diff, pct_below, graph_path=None):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()
    content = []

    # 🎓 Title
    content.append(Paragraph("Student Performance Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # 📌 Basic Details
    content.append(Paragraph(f"<b>Student ID:</b> {student_id}", styles["Normal"]))
    content.append(Paragraph(f"<b>Final Exam Score:</b> {row.get('Final_Exam_Score', 'N/A')}", styles["Normal"]))
    content.append(Paragraph(f"<b>Attendance:</b> {row.get('Attendence', 'N/A')}%", styles["Normal"]))
    content.append(Paragraph(f"<b>Study Hours:</b> {row.get('Study_hours', 'N/A')}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # 🤖 Prediction
    content.append(Paragraph("<b>Prediction</b>", styles["Heading2"]))
    content.append(Paragraph(f"Predicted Score: {predicted_score}", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # 📊 Benchmark
    content.append(Paragraph("<b>Benchmark Analysis</b>", styles["Heading2"]))
    content.append(Paragraph(f"Class Average: {class_avg}", styles["Normal"]))
    
    if benchmark_diff is not None:
        if benchmark_diff >= 0:
            content.append(Paragraph(f"You are ABOVE average by {benchmark_diff} marks", styles["Normal"]))
        else:
            content.append(Paragraph(f"You are BELOW average by {abs(benchmark_diff)} marks", styles["Normal"]))

    if pct_below is not None:
        content.append(Paragraph(f"Better than {pct_below}% of students", styles["Normal"]))

    content.append(Spacer(1, 12))

    # 📊 Performance Graph
    if graph_path and os.path.exists(graph_path):
        content.append(Paragraph("<b>Performance Visual Analysis</b>", styles["Heading2"]))
        try:
            img = Image(graph_path, width=400, height=250)
            content.append(img)
        except Exception as e:
            content.append(Paragraph(f"<i>(Graph could not be loaded: {e})</i>", styles["Normal"]))
        content.append(Spacer(1, 12))

    # 💡 Suggestions
    content.append(Paragraph("<b>Suggestions</b>", styles["Heading2"]))

    if suggestions:
        for title, tip in suggestions:
            content.append(Paragraph(f"• {title}: {tip}", styles["Normal"]))
            content.append(Spacer(1, 6))
    else:
        content.append(Paragraph("Great performance! Keep it up.", styles["Normal"]))

    # 📄 Build PDF
    doc.build(content)

    buffer.seek(0)
    return buffer

from flask import abort
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

@app.route("/teacher/student-report/<student_id>")
@login_required("teacher")
def student_report(student_id):

    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]

    if match.empty:
        abort(404, description="Student not found")

    row = match.iloc[0].to_dict()

    # ✅ Prediction
    predicted = predict_score(row)
    risk = risk_label(predicted)

    # ✅ Benchmark
    class_avg = round(float(df["Final_Exam_Score"].mean()), 1)
    benchmark_diff = round(predicted - class_avg, 1) if predicted else None
    pct_below = round(float((df["Final_Exam_Score"] <= predicted).mean() * 100), 1) if predicted else None

    # ✅ Suggestions
    suggestions = generate_suggestions(predicted, row, class_avg=class_avg) if predicted else []

    # ✅ Generate Performance Graph
    graph_path = create_performance_graph(student_id, row)

    # ✅ USE YOUR FORMATTED FUNCTION
    pdf_buf = build_student_report_pdf(
        student_id=student_id,
        row=row,
        predicted_score=predicted,
        risk=risk,
        suggestions=suggestions,
        class_avg=class_avg,
        benchmark_diff=benchmark_diff,
        pct_below=pct_below,
        graph_path=graph_path
    )

    # ✅ Cleanup temp graph image
    if graph_path and os.path.exists(graph_path):
        try:
            os.remove(graph_path)
        except:
            pass

    return send_file(
        pdf_buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"student_{student_id}_report.pdf",
    )

@app.route("/teacher/analytics")
@login_required("teacher")
def teacher_analytics():
    df = load_csv()

    # Apply filters
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]

    df = calculate_display_risk(df)

    # Basic analytics data (safe defaults)
    hist_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
    hist_counts = [0] * 10

    for score in df["display_score"]:
        idx = min(int(score // 10), 9)
        hist_counts[idx] += 1

    scatter_points = [
        {"x": row["Attendence"], "y": row["display_score"]}
        for _, row in df.iterrows()
    ]

    component_labels = ["Internal 1","Internal 2","Assignment", "Previous"]

    at_risk = df[df["risk"] == "At Risk"]
    safe = df[df["risk"] == "Safe"]

    component_at_means = [
        at_risk["internal_test 1"].mean() if not at_risk.empty else 0,
        at_risk["internal_test 2"].mean() if not at_risk.empty else 0,
        at_risk["Assignment_score"].mean() if not at_risk.empty else 0,
        at_risk["Previous_Exam_Score"].mean() if not at_risk.empty else 0
    ]

    component_safe_means = [
        safe["internal_test 1"].mean() if not safe.empty else 0,
        safe["internal_test 2"].mean() if not safe.empty else 0,
        safe["Assignment_score"].mean() if not safe.empty else 0,
        safe["Previous_Exam_Score"].mean() if not safe.empty else 0
    ]

    import json

    # Calculate basic stats for the template
    total = len(df)
    avg_score = round(df["display_score"].mean(), 1) if total > 0 else 0
    avg_att = round(df["Attendence"].mean(), 1) if total > 0 else 0
    class_avg = df.groupby("Class")["display_score"].mean().round(1).to_dict() if total > 0 else {}

    return render_template("teacher_analytics.html",
        avg_score=avg_score,
        avg_att=avg_att,
        total=total,
        class_avg=class_avg,
        classes=classes,
        sections=sections,
        sel_class=sel_class,
        sel_section=sel_section,
        hist_labels_json=json.dumps(hist_labels),
        hist_counts_json=json.dumps(hist_counts),
        scatter_points_json=json.dumps(scatter_points),
        component_labels_json=json.dumps(component_labels),
        component_at_means_json=json.dumps(component_at_means),
        component_safe_means_json=json.dumps(component_safe_means),
        attendance_perf_corr=round(df["Attendence"].corr(df["display_score"]),2) if total > 1 else 0
    )

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_message = request.json.get("message", "").lower()

        # ✅ Basic replies
        if any(word in user_message for word in ["hi", "hello", "hey"]):
            return jsonify({"reply": "👋 Hello! I am your Academic Assistant."})

        if "help" in user_message:
            return jsonify({"reply": """🤖 You can ask:
- What is my score?
- Why is my performance low?
- How to improve?
- Give suggestions"""})

        # ✅ Check login
        if "user_id" not in session:
            return jsonify({"reply": "🔒 Please login first."})

        role = session.get("role")
        
        # ✅ Load CSV for context
        try:
            df = load_csv()
        except Exception as e:
            return jsonify({"reply": f"⚠️ Error loading data file: {e}"})

        # ✅ Benchmark Data
        class_avg = round(df["Final_Exam_Score"].mean(), 1) if not df.empty else 0
        top_score = df["Final_Exam_Score"].max() if not df.empty else 0

        # ✅ Get Context based on Role
        student_id = None
        row = {}
        context_text = ""

        if role == "student" or role == "parent":
            user_record = db.get_student_by_user_id(session["user_id"]) if role == "student" else db.get_parent_by_user_id(session["user_id"])
            if user_record:
                student_id = str(user_record.get("student_code", "")).strip().upper()
        
        # If no explicit student found, but message mentions a student ID (for teachers)
        if not student_id and role == "teacher":
            import re
            match_id = re.search(r"student\s*(\d+)", user_message)
            if match_id:
                student_id = match_id.group(1)

        if student_id:
            df["Student_ID"] = df["Student_ID"].astype(str).str.strip().str.upper()
            match = df[df["Student_ID"] == student_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                score = predict_score(row)
                risk = risk_label(score)
                suggestions = generate_suggestions(score if score else 0, row)
                
                context_text = f"""
Student Profile ({student_id}):
- Predicted Score: {score}/100
- Risk Status: {risk}
- Attendance: {row.get('Attendence')}%
- Study Hours: {row.get('Study_hours')}
- Internal Marks: IT1={row.get('internal_test 1')}, IT2={row.get('internal_test 2')}, Assignment={row.get('Assignment_score')}
- Benchmarks: Class Avg is {class_avg}, Top Score is {top_score}
- Key Advice: {', '.join([tip[1] for tip in suggestions[:3]])}
"""

        # ✅ Handle History (Multi-turn)
        if "chat_history" not in session:
            session["chat_history"] = []
        
        history = session["chat_history"][-6:] # Last 3 pairs
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        # ✅ Build Professional Prompt
        prompt = f"""
You are your "AI Assistant", a world-class Academic Success Counselor.
Your goal is to provide data-driven, encouraging, and highly professional advice.

Context Information:
{context_text if context_text else "General system context: Class Average is " + str(class_avg)}

Conversation History:
{history_str}

User's Latest Message: {user_message}

Instructions:
1. If the user asks for their score or status, use the provided Profile data.
2. If no student profile is found, ask the user to provide a Student ID or log in properly.
3. Keep answers concise, professional, and helpful.
4. Use formatting (bullet points) if explaining complex tips.
5. Do not mention "Profile Data" or "Prompt" in your response. Speak naturally.
"""

        # ✅ Call Gemini API
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            reply = response.text.strip()

            if not reply:
                raise ValueError("Empty response from AI")
            
            # Save to history
            session["chat_history"].append({"role": "user", "content": user_message})
            session["chat_history"].append({"role": "bot", "content": reply})
            session.modified = True

        except Exception as e:
            print("Gemini Error:", e)
            
            # ✅ SMART FALLBACK (When AI is down)
            if student_id and 'row' in locals() and row:
                if "score" in user_message or "result" in user_message:
                    reply = f"📊 Based on your data, your predicted score is **{score}** ({risk})."
                elif "why" in user_message or "low" in user_message or "performance" in user_message:
                    reasons = explain_prediction(row)
                    reply = "📌 **Analysis of your performance:**\n" + "\n".join(reasons)
                elif "improve" in user_message or "suggest" in user_message or "help" in user_message:
                    if 'suggestions' in locals() and suggestions:
                        reply = "💡 **Here are some tips to improve:**\n" + "\n".join([f"• {tip[1]}" for tip in suggestions])
                    else:
                        reply = "👍 Your current performance is stable. Keep maintaining your attendance and study habits!"
                else:
                    reply = f"⚠️ AI Assistant is currently offline, but here is your basic status: **Score {score} ({risk})**. Please try asking about your score or improvement tips!"
            else:
                reply = "❌ I'm having trouble connecting to my AI brain right now, and I couldn't find specific student data to provide a fallback. Please try again in a moment."

        return jsonify({"reply": reply})

    except Exception as e:
        print("Chatbot Error:", e)
        return jsonify({"reply": "❌ Something went wrong. Please try again."})

@app.route("/teacher/feedback")
@login_required("teacher")
def teacher_feedback_page():
    status_filter = request.args.get("status", "open")
    feedback_list = db.get_all_feedback(status=status_filter)
    return render_template("teacher_feedback.html", 
                           feedback_list=feedback_list, 
                           current_status=status_filter)

@app.route("/teacher/reply-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def reply_feedback(fid):
    reply_text = request.form.get("reply_text", "").strip()
    if not reply_text:
        flash("Reply cannot be empty.", "warning")
        return redirect(request.referrer or url_for("teacher_feedback_page"))
    
    db.save_teacher_reply(fid, reply_text)
    flash("Reply sent and feedback marked as resolved.", "success")

    # Send email notification if possible
    fb_list = db.get_all_feedback()
    fb = next((f for f in fb_list if f["id"] == fid), None)
    if fb:
        parent_email = db.get_parent_email_by_student_code(fb["student_code"])
        if parent_email:
            subject = "Teacher Replied to Your Feedback 📩"
            content = f"""
            <h3>Hello {fb['parent_name']},</h3>
            <p>The teacher has replied to your feedback regarding student <b>{fb['student_code']}</b>.</p>
            <hr>
            <p><b>Your Message:</b> {fb['message']}</p>
            <p><b>Teacher's Reply:</b> {reply_text}</p>
            <hr>
            <p>You can view this in your dashboard.</p>
            """
            mailer.send_email_async(parent_email, subject, content)

    return redirect(request.referrer or url_for("teacher_feedback_page"))

@app.route("/teacher/resolve-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def resolve_feedback(fid):
    # Fetch feedback details before resolving to get parent info
    conn = db.get_conn()
    fb = conn.execute("SELECT pf.*, p.parent_name FROM parent_feedback pf JOIN parents p ON pf.parent_id = p.id WHERE pf.id=?", (fid,)).fetchone()
    conn.close()

    db.resolve_feedback(fid)
    flash("Feedback marked as resolved.", "success")

    if fb:
        parent_email = db.get_parent_email_by_student_code(fb["student_code"])
        if parent_email:
            subject = "Your Feedback has been Resolved ✅"
            content = mailer.get_feedback_resolved_template(fb["parent_name"], fb["student_code"])
            mailer.send_email_async(parent_email, subject, content)

    return redirect(request.referrer or url_for("teacher_feedback_page"))

@app.route("/teacher/delete-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def delete_feedback(fid):
    # This is now a "Move to Recycle Bin" action
    db.soft_delete_feedback(fid)
    flash("Message moved to Recycle Bin (Restorable for 30 days).", "warning")
    return redirect(request.referrer or url_for("teacher_feedback_page"))

@app.route("/teacher/restore-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def restore_feedback(fid):
    db.restore_feedback(fid)
    flash("Message restored successfully.", "success")
    return redirect(request.referrer or url_for("teacher_feedback_page"))

@app.route("/teacher/permanent-delete-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def permanent_delete_feedback(fid):
    # This is for manual permanent deletion from Recycle Bin
    conn = db.get_conn()
    conn.execute("DELETE FROM parent_feedback WHERE id=?", (fid,))
    conn.commit()
    conn.close()
    flash("Message deleted permanently.", "danger")
    return redirect(request.referrer or url_for("teacher_feedback_page"))
@app.route("/teacher/intervention-plan/<student_id>")
@login_required("teacher")
def intervention_plan(student_id):
    """Generate an AI-powered 4-week study roadmap PDF."""
    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]
    if match.empty:
        abort(404, description="Student not found")
    
    row = match.iloc[0].to_dict()
    score = predict_score(row)
    risk = risk_label(score)
    
    # Generate content with Gemini
    prompt = f"""
    Create a highly professional 4-week student intervention plan (academic roadmap) for a student with these stats:
    - Predicted Score: {score}/100
    - Risk Level: {risk}
    - Attendance: {row.get('Attendence')}%
    - Study Hours: {row.get('Study_hours')} hrs/day
    - Internal Test 1: {row.get('internal_test 1')}/100
    - Internal Test 2: {row.get('internal_test 2')}/100
    - Assignment Score: {row.get('Assignment_score')}/100

    The roadmap must have:
    1. A summary of current standing.
    2. Specific goals for Week 1, Week 2, Week 3, and Week 4.
    3. Daily actionable tips (e.g., 'Increase study by 1 hour', 'Focus on Internal Test 2 concepts').
    4. Motivational closing.

    Keep it concise but detailed enough to be useful. Format it clearly.
    """
    
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        roadmap_text = response.text
    except Exception as e:
        # 🛡️ PROFESSIONAL FALLBACK: Generate a high-quality static roadmap if AI fails
        print(f"AI Quota/Error: {e}")
        roadmap_text = f"""
**Summary of Standing**
Based on current data, the student has a predicted score of {score}. Immediate focus is required on attendance and internal assessment scores to stabilize performance.

**Week 1: Foundations & Attendance**
• Goal: Establish a consistent 8-hour sleep cycle and ensure 100% attendance this week.
• Tip: Review the most recent internal test papers and identify 'silly mistakes'.

**Week 2: Core Concept Reinforcement**
• Goal: Dedicate 2 hours daily to the subject with the lowest internal score.
• Tip: Create 'Concept Maps' for difficult chapters to visualize connections.

**Week 3: Practice & Assessment**
• Goal: Complete 3 previous year question papers under timed conditions.
• Tip: Focus on time management—don't spend more than 10 minutes on short answers.

**Week 4: Review & Final Prep**
• Goal: Mock exam simulation and final doubt clearing with teachers.
• Tip: Stay positive! Consistent effort in the last 3 weeks has already improved your baseline.

**Closing Note**
Consistency is the bridge between goals and accomplishment. Keep pushing forward!
"""

    # Build PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor("#7c3aed"), spaceAfter=20)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], textColor=colors.HexColor("#9333ea"), spaceBefore=15, spaceAfter=10)
    normal_style = styles['Normal']
    normal_style.leading = 14

    content = []
    content.append(Paragraph(f"Academic Intervention Plan: Student {student_id}", title_style))
    content.append(Paragraph(f"<b>Report Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')}", normal_style))
    content.append(Paragraph(f"<b>Current Status:</b> {risk} (Predicted Score: {score})", normal_style))
    content.append(Spacer(1, 15))

    # Add the roadmap text (split by lines to handle formatting)
    for line in roadmap_text.split('\n'):
        if line.strip().startswith('Week') or line.strip().startswith('**Week'):
            content.append(Paragraph(line, section_style))
        elif line.strip():
            # Basic markdown-like handling for bullet points
            clean_line = line.replace('**', '').replace('*', '•')
            content.append(Paragraph(clean_line, normal_style))
            content.append(Spacer(1, 4))

    doc.build(content)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"Intervention_Plan_{student_id}.pdf"
    )

# ════════════════════════════════════════════════════════════════════
# PARENT
# ════════════════════════════════════════════════════════════════════
@app.route("/parent")
@login_required("parent")
def parent_dashboard():
    parent = db.get_parent_by_user_id(session["user_id"])
    student = get_student_by_id(parent["student_code"])
    
    score = None
    result = None
    suggestions = []
    
    if student:
        actual = float(student.get("Final_Exam_Score", 0))
        if actual > 0:
            score = actual
        else:
            score = predict_score(student)
        result = risk_label(score)
        df = load_csv()
        class_avg = round(float(df["Final_Exam_Score"].mean()), 1) if not df.empty else 0
        suggestions = generate_suggestions(score, student, class_avg=class_avg)
        sensitivity = calculate_sensitivity(student, score)
    
    show_recycle = request.args.get("show_recycle") == "1"
    feedbacks = db.get_feedbacks_by_parent(parent["id"], exclude_deleted=not show_recycle)
    
    return render_template("parent_dashboard.html", 
                           parent=parent, student=student, 
                           score=score, result=result, suggestions=suggestions,
                           sensitivity=sensitivity, feedbacks=feedbacks,
                           show_recycle=show_recycle)

@app.route("/parent/mark-read", methods=["POST"])
@login_required("parent")
def parent_mark_read():
    parent = db.get_parent_by_user_id(session["user_id"])
    if parent:
        db.mark_feedback_as_read(parent["id"])
    return redirect(url_for("parent_dashboard"))

@app.route("/parent/log-hours", methods=["POST"])
@login_required("parent")
def parent_log_hours():
    parent = db.get_parent_by_user_id(session["user_id"])
    hours = fval(request.form, "hours")
    if hours > 0:
        db.log_study_hours(parent["student_code"], hours, f"Parent: {parent['parent_name']}")
        flash(f"Logged {hours} hours for today.", "success")
    else:
        flash("Invalid hours.", "danger")
    return redirect(url_for("parent_dashboard"))

@app.route("/parent/feedback", methods=["POST"])
@login_required("parent")
def parent_feedback():
    parent = db.get_parent_by_user_id(session["user_id"])
    msg = request.form.get("message", "").strip()
    if msg:
        sentiment = analyze_sentiment(msg)
        db.submit_feedback(parent["id"], parent["student_code"], msg, sentiment)
        flash(f"Feedback sent to teacher. (Detected Sentiment: {sentiment})", "success")
    else:
        flash("Message cannot be empty.", "danger")
    return redirect(url_for("parent_dashboard"))


@app.route("/teacher/email_alerts")
@login_required("teacher")
def email_alerts_page():
    df = load_csv()
    if not df.empty:
        df = calculate_display_risk(df)
        students = df.to_dict(orient="records")
    else:
        students = []
    
    return render_template("email_alerts.html", students=students)

def send_performance_email(student_row):
    """Helper to send a single performance report email with inline graph."""
    email = student_row.get("parent_email", "").strip()
    if not email:
        return False, "No parent email provided."
    
    sid = student_row["Student_ID"]
    score = predict_score(student_row)
    risk = risk_label(score)
    suggestions = generate_suggestions(score, student_row)
    att = student_row.get("Attendence", 0)
    
    # Generate the performance graph for the email
    # If final score is 0, use predicted score for the graph display
    temp_row = student_row.copy()
    if float(temp_row.get('Final_Exam_Score', 0)) == 0:
        temp_row['Final_Exam_Score'] = score
        
    graph_path = create_performance_graph(sid, temp_row)
    
    subject = f"Academic Performance Report: Student {sid} 📊"
    html = mailer.get_performance_report_template(sid, score, att, risk, suggestions)
    
    # Send email with inline image CID: performance_graph (referenced in template)
    mailer.send_email_async(email, subject, html, image_path=graph_path)
    
    # Note: We don't delete the graph immediately because send_email_async is async.
    # It will stay in static/temp_reports until the next time or manual cleanup.
    return True, "Email queued successfully."

@app.route("/teacher/send_email/<student_id>")
@login_required("teacher")
def send_individual_email(student_id):
    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]
    if match.empty:
        flash("Student not found.", "danger")
        return redirect(url_for("email_alerts_page"))
    
    student = match.iloc[0].to_dict()
    success, msg = send_performance_email(student)
    if success:
        flash(f"Performance report for {student_id} has been sent to {student.get('parent_email')}.", "success")
    else:
        flash(f"Error: {msg}", "danger")
    return redirect(url_for("email_alerts_page"))

@app.route("/teacher/send_all_emails")
@login_required("teacher")
def send_all_emails():
    df = load_csv()
    if df.empty:
        flash("No students found.", "warning")
        return redirect(url_for("email_alerts_page"))
    
    count = 0
    for _, row in df.iterrows():
        student = row.to_dict()
        if student.get("parent_email", "").strip():
            send_performance_email(student)
            count += 1
            
    flash(f"Success! Queued performance reports for {count} students with valid parent emails.", "success")
    return redirect(url_for("email_alerts_page"))

@app.route("/teacher/email_sample_csv")
@login_required("teacher")
def email_sample_csv():
    headers = "student_id,student_name,parent_name,parent_email\n"
    sample_row = "S1001,John Doe,Jane Doe,jane.doe@example.com"
    output = io.BytesIO()
    output.write((headers + sample_row).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="email_update_template.csv")

@app.route("/teacher/upload_email_csv", methods=["POST"])
@login_required("teacher")
def upload_email_csv():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("email_alerts_page"))
    
    try:
        df_new = pd.read_csv(file)
        df_new.columns = [c.lower().strip() for c in df_new.columns]
        
        if "student_id" not in df_new.columns or "parent_email" not in df_new.columns:
            flash("CSV must contain 'student_id' and 'parent_email' columns.", "danger")
            return redirect(url_for("email_alerts_page"))

        df_main = load_csv()
        updated_count = 0
        
        for _, row in df_new.iterrows():
            sid = str(row["student_id"]).strip()
            email = str(row["parent_email"]).strip()
            
            # Update CSV Data
            if sid in df_main["Student_ID"].astype(str).values:
                df_main.loc[df_main["Student_ID"].astype(str) == sid, "parent_email"] = email
                updated_count += 1
            
            # Update SQLite Data (Optional but good for sync)
            try:
                db.update_parent_email(sid, email)
            except:
                pass # Continue if SQLite update fails

        df_main.to_csv(DATA_FILE, index=False)
        flash(f"Successfully updated {updated_count} parent email addresses.", "success")
        
    except Exception as e:
        flash(f"Error processing CSV: {e}", "danger")
        
    return redirect(url_for("email_alerts_page"))


# ════════════════════════════════════════════════════════════════════
# AI GUIDANCE ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route("/teacher/guidance", methods=["GET", "POST"])
@login_required("teacher")
def teacher_guidance():
    student = None
    insights = None
    sid = (request.form.get("student_id") or request.args.get("student_id") or "").strip()
    if sid:
        student = get_student_by_id(sid)
        if student:
            insights = generate_advanced_insights(student)
        else:
            flash(f"Student ID {sid} not found.", "warning")
    return render_template("guidance.html", student=student, insights=insights, sid=sid, role="teacher")

@app.route("/student/guidance")
@login_required("student")
def student_guidance():
    student_record = db.get_student_by_user_id(session["user_id"])
    insights = None
    student_data = None
    if student_record:
        student_data = get_student_by_id(student_record["student_code"])
        if student_data:
            insights = generate_advanced_insights(student_data)
    return render_template("guidance.html", student=student_data, insights=insights, role="student")

@app.route("/parent/guidance")
@login_required("parent")
def parent_guidance():
    parent_record = db.get_parent_by_user_id(session["user_id"])
    insights = None
    student_data = None
    if parent_record:
        student_data = get_student_by_id(parent_record["student_code"])
        if student_data:
            insights = generate_advanced_insights(student_data)
    return render_template("guidance.html", student=student_data, insights=insights, role="parent")


# ── Career Guidance ───────────────────────────────────────────────
@app.route("/career-guidance")
def career_guidance_page():
    """Serve career guidance page for student or parent."""
    if "user_id" not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for("login_page"))
    role = session.get("role")
    if role not in ("student", "parent"):
        flash("Access denied.", "danger")
        return redirect(url_for("login_page"))

    student_data = None
    past = None
    if role == "student":
        rec = db.get_student_by_user_id(session["user_id"])
        if rec:
            student_data = get_student_by_id(rec["student_code"])
            past = db.get_latest_career_suggestion(rec["student_code"])
    else:
        rec = db.get_parent_by_user_id(session["user_id"])
        if rec:
            student_data = get_student_by_id(rec["student_code"])
            past = db.get_latest_career_suggestion(rec["student_code"])

    return render_template("career_guidance.html",
                           student=student_data, past=past, role=role)




def rule_based_career_suggestion(q, student_data):
    """Smart rule-based fallback when Gemini quota is exhausted."""
    import json

    fav = q.get("fav_subjects", "").lower()
    hobbies = q.get("hobbies", "").lower()
    work = q.get("work_style", "").lower()
    dream = q.get("career_dream", "").lower()
    higher = q.get("higher_study", "").lower()
    diploma = q.get("diploma_ok", "").lower()
    strengths = q.get("strengths", "").lower()
    weaknesses = q.get("weaknesses", "").lower()
    financial = q.get("financial", "").lower()

    score = predict_score(student_data) if student_data else 60

    # ── Stream Decision ──────────────────────────────────────────
    science_score = sum([
        "mathematics" in fav, "physics" in fav, "chemistry" in fav,
        "biology" in fav, "computer" in fav,
        "coding" in hobbies, "robotics" in hobbies,
        "doctor" in dream, "engineer" in dream, "scientist" in dream,
        "technical" in work, "research" in work,
        "logical" in strengths,
    ])
    commerce_score = sum([
        "economics" in fav, "accounts" in fav, "business" in fav,
        "business" in hobbies, "trading" in hobbies,
        "business" in dream, "entrepreneur" in dream, "ca" in dream,
        "business" in work, "leadership" in strengths,
    ])
    arts_score = sum([
        "history" in fav, "geography" in fav, "english" in fav,
        "drawing" in hobbies, "music" in hobbies, "dancing" in hobbies,
        "photography" in hobbies, "writing" in hobbies,
        "creative" in strengths, "creative" in work, "artistic" in work,
        "artist" in dream, "designer" in dream, "teacher" in dream,
    ])
    diploma_score = sum([
        "finish quickly" in higher, "working as soon" in higher,
        "very open" in diploma, "leads to" in diploma,
        "government" in financial or "scholarship" in financial,
        score < 60,
    ])

    scores_map = {
        "Science": science_score,
        "Commerce": commerce_score,
        "Arts": arts_score,
        "Diploma": diploma_score,
    }
    stream = max(scores_map, key=scores_map.get)
    if scores_map[stream] == 0:
        stream = "Science" if score >= 70 else "Commerce" if score >= 55 else "Diploma"

    # ── Career Paths by Stream ───────────────────────────────────
    careers_db = {
        "Science": [
            {"title": "Software Engineer", "description": "Design and build software systems and applications. One of the most in-demand careers in India with excellent salary prospects.", "entrance_exam": "JEE Main / KCET", "years_required": "4 years (B.E/B.Tech)"},
            {"title": "Doctor (MBBS)", "description": "Diagnose and treat patients in hospitals or private practice. Requires dedication but offers great respect and earnings.", "entrance_exam": "NEET UG", "years_required": "5.5 years (MBBS)"},
            {"title": "Data Scientist", "description": "Analyse large datasets to find patterns and help businesses make decisions. Fast-growing field in India.", "entrance_exam": "JEE / KCET + Specialisation", "years_required": "4 years (B.Tech + skills)"},
            {"title": "Mechanical Engineer", "description": "Design and manufacture machines, engines, and mechanical systems used in industry and daily life.", "entrance_exam": "JEE Main / KCET", "years_required": "4 years (B.E)"},
        ],
        "Commerce": [
            {"title": "Chartered Accountant (CA)", "description": "Handle taxation, auditing, and financial planning for companies and individuals. Highly respected profession in India.", "entrance_exam": "CA Foundation (ICAI)", "years_required": "4-5 years after 12th"},
            {"title": "Business Manager (MBA)", "description": "Lead teams and manage business operations across industries. Opens doors to top corporate roles.", "entrance_exam": "CAT / MAT / CMAT", "years_required": "5 years (BBA + MBA)"},
            {"title": "Financial Analyst", "description": "Analyse markets and investments to help companies and individuals grow their wealth.", "entrance_exam": "CFA / B.Com entrance", "years_required": "3-4 years (B.Com + certification)"},
            {"title": "Company Secretary (CS)", "description": "Ensure a company meets legal and regulatory requirements. Good career with government and private firms.", "entrance_exam": "CS Foundation (ICSI)", "years_required": "3-4 years after 12th"},
        ],
        "Arts": [
            {"title": "Graphic Designer / UI Designer", "description": "Create visual content for brands, apps, and websites. High demand in digital India.", "entrance_exam": "NID / NIFT Entrance", "years_required": "4 years (B.Des)"},
            {"title": "Journalist / Content Writer", "description": "Write and report news, stories, and content for media houses and digital platforms.", "entrance_exam": "BJMC entrance exams", "years_required": "3 years (BA Journalism)"},
            {"title": "Teacher / Educator", "description": "Teach and inspire the next generation in schools or colleges. Stable government jobs available.", "entrance_exam": "CTET / TET", "years_required": "3-4 years (BA + B.Ed)"},
            {"title": "Lawyer (LLB)", "description": "Represent clients in court and provide legal advice. Good career in criminal, civil, or corporate law.", "entrance_exam": "CLAT / LSAT", "years_required": "5 years (BA LLB integrated)"},
        ],
        "Diploma": [
            {"title": "Polytechnic Engineer", "description": "Work as a junior engineer in manufacturing, construction, or electronics. Quick entry into the workforce.", "entrance_exam": "Karnataka Polytechnic CET", "years_required": "3 years (Diploma)"},
            {"title": "ITI Technician", "description": "Skilled trade professional in electrical, mechanical, or automotive fields. Government and private jobs available.", "entrance_exam": "ITI Entrance", "years_required": "1-2 years (ITI)"},
            {"title": "Web Developer", "description": "Build websites and web applications. Can be self-taught or through a diploma course. Freelancing possible.", "entrance_exam": "None (skill-based)", "years_required": "1-2 years (Diploma/Bootcamp)"},
            {"title": "Pharmacy Technician (D.Pharm)", "description": "Assist pharmacists in dispensing medicines. Good scope in hospitals and medical shops.", "entrance_exam": "D.Pharm CET Karnataka", "years_required": "2 years (D.Pharm)"},
        ],
    }

    # ── Skill Gaps ───────────────────────────────────────────────
    skill_gaps = []
    if "mathematics" in weaknesses or score < 60:
        skill_gaps.append({"skill": "Mathematical Reasoning", "how_to_improve": "Practice 10 problems daily from RS Aggarwal or NCERT. Use Khan Academy for free video explanations."})
    if "time management" in weaknesses or "procrastination" in weaknesses:
        skill_gaps.append({"skill": "Time Management", "how_to_improve": "Use the Pomodoro technique — 25 min study, 5 min break. Make a weekly timetable and stick to it."})
    if "confidence" in weaknesses or "exam anxiety" in weaknesses:
        skill_gaps.append({"skill": "Exam Confidence", "how_to_improve": "Take mock tests regularly. Discuss doubts with teachers immediately. Practice deep breathing before exams."})
    if "language" in weaknesses or "english" in weaknesses:
        skill_gaps.append({"skill": "English Communication", "how_to_improve": "Read one English newspaper article daily. Watch English YouTube videos with subtitles. Speak in English for 10 min/day."})
    if not skill_gaps:
        skill_gaps.append({"skill": "Consistent Study Habit", "how_to_improve": "Maintain a regular 3-4 hour daily study schedule even when not near exams."})
        skill_gaps.append({"skill": "Critical Thinking", "how_to_improve": "Solve puzzle-based problems, read case studies, and question the 'why' behind every concept you learn."})
    skill_gaps.append({"skill": "Digital Literacy", "how_to_improve": "Learn basics of MS Office, email etiquette, and one productivity tool like Notion or Google Workspace."})

    # ── Colleges ─────────────────────────────────────────────────
    colleges_db = {
        "Science": [
            {"name": "RV College of Engineering, Bengaluru", "type": "Private (Autonomous)", "for_stream": "Science - Engineering"},
            {"name": "BMS College of Engineering, Bengaluru", "type": "Private (Autonomous)", "for_stream": "Science - Engineering"},
            {"name": "Mysore Medical College, Mysuru", "type": "Government", "for_stream": "Science - Medicine"},
            {"name": "NIE Institute of Technology, Mysuru", "type": "Private", "for_stream": "Science - Engineering"},
        ],
        "Commerce": [
            {"name": "Christ University, Bengaluru", "type": "Private (Deemed)", "for_stream": "Commerce / BBA / MBA"},
            {"name": "St. Joseph's College of Commerce, Bengaluru", "type": "Private (Autonomous)", "for_stream": "B.Com / Commerce"},
            {"name": "Bangalore University (affiliated colleges)", "type": "Government", "for_stream": "B.Com / BBA"},
            {"name": "ICAI Regional Office, Bengaluru", "type": "Professional Body", "for_stream": "CA Foundation"},
        ],
        "Arts": [
            {"name": "Maharaja's College, Mysuru", "type": "Government", "for_stream": "Arts / Humanities"},
            {"name": "Christ University, Bengaluru", "type": "Private (Deemed)", "for_stream": "BA / Journalism / Psychology"},
            {"name": "National Institute of Design (NID), Bengaluru", "type": "Government (Autonomous)", "for_stream": "Design / Arts"},
            {"name": "Manipal University, Manipal", "type": "Private (Deemed)", "for_stream": "Mass Communication / Arts"},
        ],
        "Diploma": [
            {"name": "Government Polytechnic, Bengaluru", "type": "Government Polytechnic", "for_stream": "Diploma Engineering"},
            {"name": "VVIET Polytechnic, Mysuru", "type": "Private Polytechnic", "for_stream": "Diploma Engineering"},
            {"name": "Government ITI, Hubballi", "type": "Government ITI", "for_stream": "Vocational / Trade"},
            {"name": "KSOU (Karnataka State Open University)", "type": "Government Open University", "for_stream": "Flexible Diploma / Degree"},
        ],
    }

    # ── Stream Reason ────────────────────────────────────────────
    reasons = {
        "Science": f"Based on your interest in {fav or 'technical subjects'} and your academic profile, Science is the best fit for you. It opens doors to engineering, medicine, and technology — some of the highest-paying fields in India. Your predicted score of {score}/100 shows you have the foundation to succeed.",
        "Commerce": f"Your interest in {fav or 'economics and business'} combined with your work preference makes Commerce the right choice. It leads to careers in CA, MBA, and finance — growing fields with excellent opportunities across India.",
        "Arts": f"Your creative interests in {hobbies or 'arts and expression'} and strength in humanities subjects make Arts the ideal stream. It offers diverse career paths in design, media, law, and education — often underrated but very rewarding.",
        "Diploma": f"Given your preference to enter the workforce quickly and your practical strengths, a Diploma or vocational course is the smartest path. It gets you job-ready in 1-3 years with strong industry demand, especially in Karnataka's manufacturing and tech sectors.",
    }

    notes = {
        "Science": "Remember, choosing Science doesn't mean you must become an engineer or doctor — it keeps all options open. Focus on building your basics in Maths and Science now, and the right career will follow. You have great potential!",
        "Commerce": "Commerce is not just about accounts — it's about understanding how the world works economically. Stay consistent, explore internships early, and don't be afraid to take up business competitions. You're on a great path!",
        "Arts": "Arts students often become the most creative and impactful professionals. Don't let anyone underestimate your stream — designers, lawyers, journalists, and educators shape society. Believe in your unique strengths!",
        "Diploma": "A diploma is not a compromise — it's a smart, practical choice that gets you earning faster than most. Many successful engineers and entrepreneurs started with a diploma. Stay focused and keep upgrading your skills!",
    }

    return {
        "recommended_stream": stream,
        "stream_reason": reasons[stream],
        "career_paths": careers_db[stream],
        "skill_gaps": skill_gaps[:4],
        "colleges_to_explore": colleges_db[stream],
        "motivational_note": notes[stream],
    }


@app.route("/api/career-suggest", methods=["POST"])
def career_suggest_api():
    """Generate AI career suggestion from questionnaire + marks."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401

    role = session.get("role")
    data = request.get_json(force=True)

    # Resolve student
    student_code = None
    student_data = None
    if role == "student":
        rec = db.get_student_by_user_id(session["user_id"])
        if rec:
            student_code = rec["student_code"]
            student_data = get_student_by_id(student_code)
    elif role == "parent":
        rec = db.get_parent_by_user_id(session["user_id"])
        if rec:
            student_code = rec["student_code"]
            student_data = get_student_by_id(student_code)

    # Build marks context
    marks_ctx = ""
    if student_data:
        score = predict_score(student_data)
        marks_ctx = f"""
Student Academic Profile:
- Predicted Final Score: {score}/100  ({risk_label(score)})
- Attendance: {student_data.get('Attendence', 'N/A')}%
- Study Hours/Day: {student_data.get('Study_hours', 'N/A')}
- Internal Test 1: {student_data.get('internal_test 1', 'N/A')}
- Internal Test 2: {student_data.get('internal_test 2', 'N/A')}
- Assignment Score: {student_data.get('Assignment_score', 'N/A')}
- Previous Exam Score: {student_data.get('Previous_Exam_Score', 'N/A')}
- Final Exam Score: {student_data.get('Final_Exam_Score', 'N/A')}
"""

    # Build questionnaire context
    q = data.get("questionnaire", {})
    q_ctx = f"""
Student Questionnaire Responses:
- Favourite subjects: {q.get('fav_subjects', 'Not specified')}
- Disliked subjects: {q.get('disliked_subjects', 'Not specified')}
- Hobbies / Interests: {q.get('hobbies', 'Not specified')}
- Preferred work style: {q.get('work_style', 'Not specified')}
- Career dream / aspiration: {q.get('career_dream', 'Not specified')}
- Willing to study 4+ more years (degree)?: {q.get('higher_study', 'Not specified')}
- Interested in vocational/diploma?: {q.get('diploma_ok', 'Not specified')}
- Strengths (self-assessed): {q.get('strengths', 'Not specified')}
- Weaknesses (self-assessed): {q.get('weaknesses', 'Not specified')}
- Family financial background (optional): {q.get('financial', 'Not specified')}
"""

    prompt = f"""You are an expert Indian high school career counsellor with 20 years of experience.
A student has completed an interest questionnaire and their academic marks are available.
Based on BOTH the questionnaire AND the academic data below, give a thorough, personalised career guidance report.

{marks_ctx}
{q_ctx}

Respond ONLY with a valid JSON object (no markdown, no extra text) with this exact structure:
{{
  "recommended_stream": "Science / Commerce / Arts / Diploma / Vocational",
  "stream_reason": "2-3 sentence explanation why this stream fits them",
  "career_paths": [
    {{"title": "Career title", "description": "2 sentence description", "entrance_exam": "Exam name or None", "years_required": "X years after 10th"}}
  ],
  "skill_gaps": [
    {{"skill": "Skill name", "how_to_improve": "Short actionable advice"}}
  ],
  "colleges_to_explore": [
    {{"name": "College / Institute name", "type": "Government/Private/Polytechnic", "for_stream": "Stream"}}
  ],
  "motivational_note": "A warm, encouraging 2-3 sentence personalised message to the student"
}}

Include 3-4 career paths, 3-4 skill gaps, and 3-4 colleges. Make it India-specific and realistic.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()
        # Strip markdown fences if any
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json
        result = json.loads(raw)

        # Save to DB
        if student_code:
            import json as _json
            db.save_career_suggestion(
                student_code,
                _json.dumps(q),
                _json.dumps(result)
            )

        return jsonify({"success": True, "result": result})
    except Exception as e:
        traceback.print_exc()
        # ── Fallback: use rule-based engine if Gemini quota exhausted ──
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
            import json as _json
            result = rule_based_career_suggestion(q, student_data)
            if student_code:
                db.save_career_suggestion(
                    student_code,
                    _json.dumps(q),
                    _json.dumps(result)
                )
            return jsonify({"success": True, "result": result, "fallback": True})
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)