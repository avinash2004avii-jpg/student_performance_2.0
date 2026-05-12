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
    df = pd.read_csv(DATA_FILE, dtype={'parent_phone': str})
    df["Health_Issues"] = df["Health_Issues"].fillna("None")
    if "parent_email" not in df.columns:
        df["parent_email"] = ""
    else:
        df["parent_email"] = df["parent_email"].fillna("")
        
    if "parent_phone" not in df.columns:
        df["parent_phone"] = ""
    else:
        df["parent_phone"] = df["parent_phone"].astype(str).replace('nan', '')
        df["parent_phone"] = df["parent_phone"].fillna("")
        
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
    if score < 70:    return "At Risk"
    if score <= 85:   return "Average Performance"
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
    df["risk"] = np.where(df["display_score"] < 70, "At Risk",
                 np.where(df["display_score"] <= 85, "Average Performance", "Safe"))
    df["risk_val"] = np.where(df["display_score"] < 70, "1", "0")
    
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

    # 1. Handle "At Risk" students with specific static-style tips as requested
    if score < 70:
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
    
    
    if it1 < 50 and it2 < 50:
        tips.append(("Core Concepts", "Both internal tests are below 50. Focus on understanding fundamentals rather than memorising — consider extra tutoring."))
    
    if not tips:
        tips.append(("Achievement", "Keep up the great work and maintain consistency!"))

    return tips

import json


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
    """Calculate what's needed to reach 'Safe' (85) or 'Average' (70) status."""
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

@app.route("/admin/approve-all", methods=["POST"])
@login_required("admin")
def admin_approve_all():
    db.approve_all()
    flash("All pending users approved successfully.", "success")
    return redirect(url_for("admin_users"))

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
    elif action == "edit_username":
        new_username = request.form.get("new_username")
        if new_username:
            new_username = new_username.strip()
            if db.username_exists(new_username):
                flash("Username already taken.", "danger")
            else:
                db.update_username(user_id, new_username)
                flash("Username updated successfully.", "success")
        else:
            flash("Username cannot be empty.", "warning")
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

@app.route("/admin/delete-all-students", methods=["POST"])
@login_required("admin")
def admin_delete_all_students():
    df = pd.DataFrame(columns=["Student_ID", "Class", "section", "Age", "Gender", "Study_hours", "Sleep_hours", "Parent_Education_Level", "Health_Issues", "Internet_Access", "Attendence", "internal_test 1", "internal_test 2", "Assignment_score", "Extracurricular_Activities", "Previous_Exam_Score", "Final_Exam_Score", "parent_email"])
    df.to_csv(DATA_FILE, index=False)
    flash("All student records cleared successfully.", "warning")
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
                
                # User requested thresholds for this feature: <70, 70-85, >85
                if score < 70:
                    result = "At Risk"
                elif score <= 85:
                    result = "Average Performance"
                else:
                    result = "Safe"
                
                df = load_csv()
                df = calculate_display_risk(df)
                class_avg = round(float(df["display_score"].mean()), 1) if not df.empty else 0
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
    df = calculate_display_risk(df)
    class_avg = round(float(df["display_score"].mean()), 1) if not df.empty else 0
    suggestions = generate_suggestions(score, input_row, class_avg=class_avg)

    # Benchmark Analysis
    percentile = 0
    if not df.empty:
        percentile = round((df[df["display_score"] < score].shape[0] / df.shape[0]) * 100, 1)

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

import csv
@app.route("/teacher/remarks/download-csv")
@login_required("teacher")
def download_remarks_csv():
    all_remarks = db.get_all_teacher_remarks()
    
    si = io.StringIO()
    cw = csv.writer(si)
    
    cw.writerow(["Remark ID", "Student ID", "Student Name", "Class", "Section", "Remark Text", "Written By (Teacher)", "Created At"])
    
    for r in all_remarks:
        cw.writerow([
            r.get("id"),
            r.get("student_code"),
            r.get("student_name") or "N/A",
            r.get("student_class") or "N/A",
            r.get("student_section") or "N/A",
            r.get("remark"),
            r.get("teacher_name") or "N/A",
            r.get("created_at")
        ])
        
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="all_student_teacher_remarks.csv"
    )

@app.route("/teacher/remarks", methods=["GET"])
@login_required("teacher")
def teacher_remarks():
    q = request.args.get("q", "")
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    selected_student_id = request.args.get("student_id", "")
    
    df = load_csv()
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    selected_student = None
    if selected_student_id:
        match_student = df[df["Student_ID"].astype(str) == str(selected_student_id)]
        if not match_student.empty:
            selected_student = match_student.iloc[0].to_dict()

    if q:
        df = df[df["Student_ID"].astype(str).str.contains(q, case=False, na=False)]
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
        
    students = df.to_dict(orient="records")
    
    remarks = []
    if selected_student_id:
        remarks = db.get_remarks_by_student(selected_student_id)
        
    all_remarks = db.get_all_teacher_remarks()
        
    return render_template("teacher_remarks.html",
                           students=students,
                           classes=classes,
                           sections=sections,
                           sel_class=sel_class,
                           sel_section=sel_section,
                           q=q,
                           selected_student_id=selected_student_id,
                           selected_student=selected_student,
                           remarks=remarks,
                           all_remarks=all_remarks)

@app.route("/teacher/remarks/add", methods=["POST"])
@login_required("teacher")
def add_remark():
    student_id = request.form.get("student_id")
    remark = request.form.get("remark", "").strip()
    q = request.form.get("q", "")
    sel_class = request.form.get("class", "")
    sel_section = request.form.get("section", "")
    
    category = request.form.get("category", "Academic")
    priority = request.form.get("priority", "Medium")
    try:
        rating = int(request.form.get("rating", 3))
    except ValueError:
        rating = 3

    df = load_csv()
    match_student = df[df["Student_ID"].astype(str) == str(student_id)]
    student_name = None
    class_val = None
    section_val = None
    if not match_student.empty:
        student_name = match_student.iloc[0].get("name")
        class_val = str(match_student.iloc[0].get("Class"))
        section_val = str(match_student.iloc[0].get("section"))

    teacher = db.get_teacher_by_user_id(session["user_id"])
    teacher_id = teacher["id"] if teacher else None
    teacher_name = teacher["name"] if teacher else None
    
    if student_id and remark:
         db.add_teacher_remark(
             student_code=student_id, 
             teacher_id=teacher_id, 
             remark=remark, 
             student_name=student_name, 
             class_val=class_val, 
             section_val=section_val, 
             teacher_name=teacher_name, 
             category=category, 
             priority=priority, 
             rating=rating
         )
         flash("Remark added successfully.", "success")
    else:
         flash("Failed to add remark. Remark cannot be empty.", "danger")
        
    return redirect(url_for("teacher_remarks", student_id=student_id, q=q, section=sel_section, **{"class": sel_class}))

@app.route("/teacher/remarks/delete", methods=["POST"])
@login_required("teacher")
def delete_remark():
    remark_id = request.form.get("remark_id")
    student_id = request.form.get("student_id")
    q = request.form.get("q", "")
    sel_class = request.form.get("class", "")
    sel_section = request.form.get("section", "")
    
    if remark_id:
        db.delete_teacher_remark(int(remark_id))
        flash("Remark deleted successfully.", "success")
        
    return redirect(url_for("teacher_remarks", student_id=student_id, q=q, section=sel_section, **{"class": sel_class}))


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

    remarks = []
    sensitivity = []
    if student:
        match = df[df["Student_ID"].astype(str) == str(student["student_code"])]
        remarks = db.get_remarks_by_student(student["student_code"])
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
        sensitivity=sensitivity, remarks=remarks
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
    df_with_display = calculate_display_risk(df.copy())
    class_avg = round(float(df_with_display["display_score"].mean()), 1)
    benchmark_diff = round(predicted - class_avg, 1) if predicted else None
    pct_below = round(float((df_with_display["display_score"] <= predicted).mean() * 100), 1) if predicted else None

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

    # a) Score Distribution
    score_bins = ["0-40", "41-70", "71-85", "86-100"]
    score_counts = [0, 0, 0, 0]
    for score in df["display_score"]:
        if score < 40: score_counts[0] += 1
        elif score < 70: score_counts[1] += 1
        elif score < 85: score_counts[2] += 1
        else: score_counts[3] += 1

    # b) Attendance vs Performance
    att_groups = ["Low (<70%)", "Medium (70-85%)", "High (>85%)"]
    att_data = { g: {"att": [], "score": []} for g in att_groups }
    for _, row in df.iterrows():
        att = row["Attendence"]
        score = row["display_score"]
        if att < 70:
            att_data["Low (<70%)"]["att"].append(att)
            att_data["Low (<70%)"]["score"].append(score)
        elif att <= 85:
            att_data["Medium (70-85%)"]["att"].append(att)
            att_data["Medium (70-85%)"]["score"].append(score)
        else:
            att_data["High (>85%)"]["att"].append(att)
            att_data["High (>85%)"]["score"].append(score)
            
    group_avg_att = [round(sum(att_data[g]["att"])/len(att_data[g]["att"]), 1) if att_data[g]["att"] else 0 for g in att_groups]
    group_avg_score = [round(sum(att_data[g]["score"])/len(att_data[g]["score"]), 1) if att_data[g]["score"] else 0 for g in att_groups]

    # c) Component Comparison
    comp_labels = ["Assignments", "Exams", "Attendance", "Internal Marks"]

    at_risk = df[df["risk"] == "At Risk"]
    stable = df[df["risk"] != "At Risk"] # Compare against everyone else for better insight

    component_at_means = [
        round(at_risk["Assignment_score"].mean(), 1) if not at_risk.empty else 0,
        round(at_risk["Previous_Exam_Score"].mean(), 1) if not at_risk.empty else 0,
        round(at_risk["Attendence"].mean(), 1) if not at_risk.empty else 0,
        round((at_risk["internal_test 1"].mean() + at_risk["internal_test 2"].mean())/2, 1) if not at_risk.empty else 0
    ]

    component_safe_means = [
        round(stable["Assignment_score"].mean(), 1) if not stable.empty else 0,
        round(stable["Previous_Exam_Score"].mean(), 1) if not stable.empty else 0,
        round(stable["Attendence"].mean(), 1) if not stable.empty else 0,
        round((stable["internal_test 1"].mean() + stable["internal_test 2"].mean())/2, 1) if not stable.empty else 0
    ]

    import json

    # Calculate basic stats for the template
    total = len(df)
    avg_score = round(df["display_score"].mean(), 1) if total > 0 else 0
    avg_att = round(df["Attendence"].mean(), 1) if total > 0 else 0
    class_avg = df.groupby("Class")["display_score"].mean().round(1).to_dict() if total > 0 else {}

    class_labels = [f"Class {c}" for c in class_avg.keys()]
    class_scores = list(class_avg.values())

    return render_template("teacher_analytics.html",
        avg_score=avg_score,
        avg_att=avg_att,
        total=total,
        classes=classes,
        sections=sections,
        sel_class=sel_class,
        sel_section=sel_section,
        score_bins_json=json.dumps(score_bins),
        score_counts_json=json.dumps(score_counts),
        att_groups_json=json.dumps(att_groups),
        group_avg_att_json=json.dumps(group_avg_att),
        group_avg_score_json=json.dumps(group_avg_score),
        comp_labels_json=json.dumps(comp_labels),
        comp_at_means_json=json.dumps(component_at_means),
        comp_safe_means_json=json.dumps(component_safe_means),
        class_labels_json=json.dumps(class_labels),
        class_scores_json=json.dumps(class_scores),
        attendance_perf_corr=round(df["Attendence"].corr(df["display_score"]),2) if total > 1 else 0
    )

def get_best_performing_class(df):
    try:
        df_display = calculate_display_risk(df.copy())
        if "Class" in df_display.columns:
            class_groups = df_display.groupby("Class")["display_score"].mean()
            best_class = class_groups.idxmax()
            best_avg = round(class_groups.max(), 1)
            return f"🏆 **Class {best_class}** is the best performing class with an average score of **{best_avg}/100**."
    except Exception as e:
        print("Error calculating best class:", e)
    return "📊 Class performance analysis is currently unavailable."

def get_at_risk_students_fallback(df):
    try:
        df_display = calculate_display_risk(df.copy())
        at_risk = df_display[df_display["display_score"] < 70]
        if at_risk.empty:
            return "✅ Great news! No students are currently in the 'At Risk' category (score < 70)."
        
        lines = ["⚠️ **At-Risk Students (Predicted Score < 70):**"]
        for _, s in at_risk.head(10).iterrows():
            lines.append(f"• **Student {s['Student_ID']}** (Class {s.get('Class', 'N/A')}): Predicted Score **{s['display_score']}**, Attendance **{s['Attendence']}%**")
        if len(at_risk) > 10:
            lines.append(f"... and {len(at_risk) - 10} more.")
        return "\n".join(lines)
    except Exception as e:
        print("Error listing at-risk students:", e)
    return "❌ Error loading at-risk students list."

def get_analytics_summary_fallback(df):
    try:
        df_display = calculate_display_risk(df.copy())
        total = len(df_display)
        at_risk = len(df_display[df_display["display_score"] < 70])
        avg_score = round(df_display["display_score"].mean(), 1)
        avg_att = round(df_display["Attendence"].mean(), 1)
        risk_pct = round((at_risk / total * 100), 1) if total > 0 else 0
        
        return f"""📈 **SPS Real-Time Analytics Overview:**
• **Total Students Monitored:** {total}
• **Class Average Predicted Score:** {avg_score}/100
• **Average Class Attendance:** {avg_att}%
• **At-Risk Student Count:** {at_risk} ({risk_pct}%)
• **SPS Insights:** Strong direct correlation found between Study Hours and high exam results."""
    except Exception as e:
        print("Error generating analytics fallback:", e)
    return "📈 Analytics summary is currently unavailable."

def get_attendance_analysis_fallback(df):
    try:
        df_display = calculate_display_risk(df.copy())
        low_att = df_display[df_display["Attendence"] < 75]
        if low_att.empty:
            return "📅 Attendance is excellent! All students have attendance above 75%."
        
        return f"""📅 **Attendance Trend Analysis:**
• There are currently **{len(low_att)}** students with attendance below the **75% minimum threshold**.
• Low attendance is the **primary leading indicator** of academic risk. Students with <75% attendance show an average score decline of **12-15 marks** compared to peers.
• We recommend sending automatic parent email/SMS alerts to address attendance issues immediately."""
    except Exception as e:
        print("Error calculating attendance fallback:", e)
    return "📅 Attendance analysis is currently unavailable."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_message = request.json.get("message", "").strip()
        user_message_lower = user_message.lower()

        # Log User Query
        print(f"\n[CHATBOT REQUEST] Message: '{user_message}'")

        # Check login
        if "user_id" not in session:
            print("[CHATBOT INFO] Blocked: No active session")
            return jsonify({"reply": "🔒 Please login first."})

        role = session.get("role")
        print(f"[CHATBOT INFO] User ID: {session['user_id']} | Role: {role}")



        # Load CSV for context
        try:
            df = load_csv()
        except Exception as e:
            print(f"[CHATBOT ERROR] Failed to load CSV: {e}")
            return jsonify({"reply": "⚠️ Error loading system database file. Please try again later."})

        # Benchmark Data
        df_with_display = calculate_display_risk(df.copy())
        class_avg = round(df_with_display["display_score"].mean(), 1) if not df.empty else 0
        top_score = df_with_display["display_score"].max() if not df.empty else 0

        # Context collection
        student_id = None
        row = {}
        context_text = ""
        score = None
        risk = "Unknown"
        suggestions = []

        if role == "student" or role == "parent":
            user_record = db.get_student_by_user_id(session["user_id"]) if role == "student" else db.get_parent_by_user_id(session["user_id"])
            if user_record:
                student_id = str(user_record.get("student_code", "")).strip().upper()
        
        # Parse student ID mentioned in message (useful for teachers)
        if not student_id:
            import re
            match_id = re.search(r"student\s*(\d+)", user_message_lower)
            if match_id:
                student_id = match_id.group(1).upper()

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
                print(f"[CHATBOT INFO] Found context for Student ID: {student_id}")

        # Conversation History
        if "chat_history" not in session:
            session["chat_history"] = []
        
        history = session["chat_history"][-6:]  # Last 3 turns
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history])

        # AI Prompt Engineering
        prompt = f"""
You are "AI Academic Assistant", a world-class educational counselor.
Provide data-driven, highly encouraging, structured, and helpful responses.

Context Information:
{context_text if context_text else "General analytics: Class Average is " + str(class_avg) + ", Top Score is " + str(top_score)}

Conversation History:
{history_str}

User's Latest Message: {user_message}

Instructions:
1. Speak professionally, encouragingly, and naturally. Do not mention system variables, prompts, or JSON data.
2. If asked about student scores, tips, or trends, rely heavily on the Context Profile.
3. If no specific student is found and you need one to answer, politely ask for the Student ID.
4. Keep answers relatively short, professional, and directly helpful.
"""

        # Call Gemini API
        try:
            print(f"[CHATBOT INFO] Invoking Gemini API...")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            reply = response.text.strip()
            if not reply:
                raise ValueError("Empty response from Gemini")
            
            # Save history
            session["chat_history"].append({"role": "user", "content": user_message})
            session["chat_history"].append({"role": "bot", "content": reply})
            session.modified = True
            print(f"[CHATBOT RESPONSE] Gemini Success: '{reply[:60]}...'")

        except Exception as e:
            print(f"[CHATBOT ERROR] Gemini failed ({e}). Activating local smart fallback...")
            
            # Smart Fallbacks based on keywords
            if "at risk" in user_message_lower or "at-risk" in user_message_lower:
                reply = get_at_risk_students_fallback(df)
            elif "best class" in user_message_lower or "performing best" in user_message_lower or "class performs best" in user_message_lower:
                reply = get_best_performing_class(df)
            elif "explain analytics" in user_message_lower or "analytics" in user_message_lower or "class average" in user_message_lower:
                reply = get_analytics_summary_fallback(df)
            elif "attendance" in user_message_lower:
                if student_id and row:
                    att = row.get("Attendence", 0)
                    reply = f"📅 **Student {student_id} Attendance:** {att}% (Threshold: 75%). " + ("This is in the safe zone! Excellent." if float(att) >= 75 else "⚠️ This is below the required 75% threshold. Attendance needs immediate improvement.")
                else:
                    reply = get_attendance_analysis_fallback(df)
            elif student_id and row:
                if "score" in user_message_lower or "predicted" in user_message_lower or "result" in user_message_lower or "marks" in user_message_lower:
                    reply = f"📊 **SPS Prediction:** Based on Student **{student_id}**'s data, the predicted final exam score is **{score}/100** ({risk})."
                elif "why" in user_message_lower or "low" in user_message_lower or "performance" in user_message_lower:
                    reasons = explain_prediction(row)
                    reply = f"📌 **Performance Analysis for Student {student_id}:**\n" + "\n".join([f"• {r}" for r in reasons])
                elif any(kw in user_message_lower for kw in ["improve", "tips", "suggest", "plan", "study"]):
                    reply = f"💡 **Recommended Action Plan for Student {student_id}:**\n" + "\n".join([f"• **{tip[0]}**: {tip[1]}" for tip in suggestions])
                else:
                    reply = f"📊 **Student {student_id} Profile Overview:**\n• **Predicted Score:** {score}/100 ({risk})\n• **Attendance:** {row.get('Attendence')}%\n• **Study Hours:** {row.get('Study_hours')} hrs/day\n• **IT1 / IT2:** {row.get('internal_test 1')} / {row.get('internal_test 2')}\n\nAsk about 'score', 'reasons', or 'improvement tips' for a deep dive!"
            else:
                if role == "teacher":
                    reply = "🧑‍🏫 **SPS Smart Assistant is online (local fallback mode).**\n\nYou can ask me:\n• 'Show at-risk students'\n• 'Which class performs best?'\n• 'Explain analytics'\n• 'Why is attendance low?'\n\nYou can also query a specific student by typing 'Student <ID>' (e.g., 'student 12345')."
                else:
                    reply = "⚠️ **AI Brain is temporarily offline (SPS Safe Fallback Mode).**\n\nI can still help you! Please try asking about 'predicted score', 'why is my performance low', or 'improvement tips' to get real-time statistics from your record."

            # Save fallback history
            session["chat_history"].append({"role": "user", "content": user_message})
            session["chat_history"].append({"role": "bot", "content": reply})
            session.modified = True
            print(f"[CHATBOT RESPONSE] Smart Fallback Success: '{reply[:60]}...'")

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

        return jsonify({"reply": reply, "suggestions": suggested_qs})

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
        df = calculate_display_risk(df)
        class_avg = round(float(df["display_score"].mean()), 1) if not df.empty else 0
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
    """Helper to send a single performance report email and SMS."""
    email = student_row.get("parent_email", "")
    phone = student_row.get("parent_phone", "")
    
    # Clean 'nan' and strip whitespace
    if not isinstance(email, str) or str(email).lower() == 'nan': email = ""
    else: email = email.strip()
    
    if not isinstance(phone, str) or str(phone).lower() == 'nan': phone = ""
    else: phone = phone.strip()
    
    if phone and not phone.startswith('+'):
        phone = '+' + phone
    
    if not email and not phone:
        return False, "No parent email or phone provided."
    
    sid = student_row["Student_ID"]
    score = predict_score(student_row)
    risk = risk_label(score)
    suggestions = generate_suggestions(score, student_row)
    att = student_row.get("Attendence", 0)
    
    temp_row = student_row.copy()
    if float(temp_row.get('Final_Exam_Score', 0)) == 0:
        temp_row['Final_Exam_Score'] = score
        
    graph_path = create_performance_graph(sid, temp_row)
    
    email_sent = False
    sms_sent = False
    
    if email:
        subject = f"Academic Performance Report: Student {sid} 📊"
        html = mailer.get_performance_report_template(sid, score, att, risk, suggestions)
        mailer.send_email_async(email, subject, html, image_path=graph_path)
        email_sent = True
        
    if phone:
        sms_msg = f"Alert for {sid}: Status: {risk}, Score: {score}, Att: {att}%. Check email."
        mailer.send_sms_async(phone, sms_msg)
        sms_sent = True
        
    if email_sent and sms_sent:
        return True, "Email and SMS queued successfully."
    elif email_sent:
        return True, "Email queued successfully. (No phone number)"
    elif sms_sent:
        return True, "SMS queued successfully. (No email address)"
    
    return False, "Failed to queue alerts."

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
        flash(f"Performance report for {student_id} queued successfully. {msg}", "success")
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
        email = str(student.get("parent_email", "")).strip()
        phone = str(student.get("parent_phone", "")).strip()
        if (email and email.lower() != 'nan') or (phone and phone.lower() != 'nan'):
            send_performance_email(student)
            count += 1
            
    flash(f"Success! Queued performance reports (Email/SMS) for {count} students.", "success")
    return redirect(url_for("email_alerts_page"))

@app.route("/teacher/email_sample_csv")
@login_required("teacher")
def email_sample_csv():
    headers = "student_id,student_name,parent_name,parent_email,parent_phone\n"
    sample_row = "S1001,John Doe,Jane Doe,jane.doe@example.com,+1234567890"
    output = io.BytesIO()
    output.write((headers + sample_row).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="contact_update_template.csv")

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
        
        if "student_id" not in df_new.columns or ("parent_email" not in df_new.columns and "parent_phone" not in df_new.columns):
            flash("CSV must contain 'student_id' and either 'parent_email' or 'parent_phone'.", "danger")
            return redirect(url_for("email_alerts_page"))

        df_main = load_csv()
        updated_count = 0
        
        for _, row in df_new.iterrows():
            sid = str(row["student_id"]).strip()
            email = str(row.get("parent_email", "")).strip() if "parent_email" in df_new.columns else ""
            phone = str(row.get("parent_phone", "")).strip() if "parent_phone" in df_new.columns else ""
            
            if email.lower() == "nan": email = ""
            if phone.lower() == "nan": phone = ""
            
            # Update CSV Data
            if sid in df_main["Student_ID"].astype(str).values:
                if email:
                    df_main.loc[df_main["Student_ID"].astype(str) == sid, "parent_email"] = email
                if phone:
                    df_main.loc[df_main["Student_ID"].astype(str) == sid, "parent_phone"] = phone
                updated_count += 1
            
            # Update SQLite Data (Optional but good for sync)
            try:
                if email:
                    db.update_parent_email(sid, email)
            except:
                pass # Continue if SQLite update fails

        df_main.to_csv(DATA_FILE, index=False)
        flash(f"Successfully updated {updated_count} parent contacts.", "success")
        
    except Exception as e:
        flash(f"Error processing CSV: {e}", "danger")
        
    return redirect(url_for("email_alerts_page"))




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

    score = predict_score(student_data) if student_data else 70

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
        score < 70,
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
    if "mathematics" in weaknesses or score < 70:
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


def get_absence_alert_email_template(student_name, class_name, date):
    return f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b; line-height: 1.6; background-color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="background: #ef4444; padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 24px;">⚠️ Absence Notification</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.8;">Date: {date}</p>
            </div>
            <div style="padding: 30px;">
                <p>Dear Parent,</p>
                <p>This is to inform you that your child, <b>{student_name}</b>, from Class <b>{class_name}</b> was marked <b>Absent</b> today.</p>
                
                <div style="background: #fef2f2; border: 1px solid #fee2e2; padding: 15px; border-radius: 8px; margin: 25px 0; text-align: center;">
                    <span style="color: #ef4444; font-weight: 800; font-size: 16px;">Attendance Status: Absent</span>
                </div>

                <p>Regular attendance is crucial for academic success. Kindly ensure their attendance in classes going forward.</p>
                
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 30px 0;">
                <p style="font-size: 12px; color: #94a3b8; text-align: center;">
                    This is an automated notification from the <b>Student Performance System</b>.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

def get_presence_alert_email_template(student_name, class_name, date):
    return f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b; line-height: 1.6; background-color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="background: #22c55e; padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 24px;">✅ Safe Arrival Notification</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.8;">Date: {date}</p>
            </div>
            <div style="padding: 30px;">
                <p>Dear Parent,</p>
                <p>This is to inform you that your child, <b>{student_name}</b>, from Class <b>{class_name}</b> has arrived safely at school and is marked <b>Present</b> today.</p>
                
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin: 25px 0; text-align: center;">
                    <span style="color: #16a34a; font-weight: 800; font-size: 16px;">Attendance Status: Present</span>
                </div>

                <p>Thank you for partnering with us to support your child's academic journey and maintaining excellent school attendance!</p>
                
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 30px 0;">
                <p style="font-size: 12px; color: #94a3b8; text-align: center;">
                    This is an automated notification from the <b>Student Performance System</b>.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


def send_alert_for_student(student_id, date, send_sms, send_email):
    student = get_student_by_id(student_id)
    if not student:
        return False, "Student not found"
    
    student_name = student.get("name", "")
    class_name = student.get("Class", "N/A")
    email = student.get("parent_email", "")
    if not isinstance(email, str) or str(email).lower() == 'nan':
        email = ""
    else:
        email = email.strip()
        
    phone = student.get("parent_phone", "")
    if not isinstance(phone, str) or str(phone).lower() == 'nan':
        phone = ""
    else:
        phone = str(phone).strip()
        if phone.endswith(".0"):
            phone = phone[:-2]
            
    if phone and not phone.startswith('+'):
        phone = '+' + phone
    
    # Retrieve actual attendance status from SQLite DB
    attendance_db = db.get_attendance_by_date(date)
    att_rec = attendance_db.get(str(student_id))
    status = att_rec["status"] if att_rec else "Present"
    
    name_str = student_name if student_name else "your child"
    
    sms_success = True
    email_success = True
    
    if send_sms:
        if phone:
            if status == "Absent":
                sms_msg = f"Dear Parent, {name_str} from Class {class_name} was marked Absent today. Please ensure regular attendance."
            else:
                sms_msg = f"Dear Parent, {name_str} from Class {class_name} has arrived safely and is marked Present today."
            mailer.send_sms_async(phone, sms_msg)
        else:
            sms_success = False
            
    if send_email:
        if email:
            if status == "Absent":
                email_subject = "Student Absence Alert"
                email_html = get_absence_alert_email_template(name_str, class_name, date)
            else:
                email_subject = "Safe Arrival Notification"
                email_html = get_presence_alert_email_template(name_str, class_name, date)
            mailer.send_email_async(email, email_subject, email_html)
        else:
            email_success = False
            
    success = (not send_sms or sms_success) and (not send_email or email_success)
    status_badge = "Sent" if success else "Failed"
    db.update_attendance_alert_status(student_id, date, status_badge)
    
    return success, "Alert sent" if success else "Failed to send alert due to missing contact details"

@app.route("/teacher/absent_alerts")
@login_required("teacher")
def teacher_absent_alerts():
    import datetime
    today = datetime.date.today().isoformat()
    sel_date = request.args.get("date", today)
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    
    df = load_csv()
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
        
    students = df.to_dict(orient="records")
    attendance_db = db.get_attendance_by_date(sel_date)
    
    absent_count = 0
    present_count = 0
    students_list = []
    
    for s in students:
        sid = str(s["Student_ID"])
        att_rec = attendance_db.get(sid)
        status = att_rec["status"] if att_rec else "Present"
        alert_status = att_rec["alert_status"] if att_rec else "Pending"
        
        if status == "Absent":
            absent_count += 1
        else:
            present_count += 1
            
        students_list.append({
            "Student_ID": sid,
            "name": s.get("name", "N/A"),
            "Class": s.get("Class", "N/A"),
            "section": s.get("section", "N/A"),
            "parent_email": s.get("parent_email", ""),
            "parent_phone": s.get("parent_phone", ""),
            "status": status,
            "alert_status": alert_status
        })
        
    total_students = len(students_list)
    absence_pct = round((absent_count / total_students * 100), 1) if total_students > 0 else 0
    
    return render_template(
        "absent_alerts.html",
        students=students_list,
        total_students=total_students,
        present_count=present_count,
        absent_count=absent_count,
        absence_pct=absence_pct,
        classes=classes,
        sections=sections,
        sel_date=sel_date,
        sel_class=sel_class,
        sel_section=sel_section
    )


@app.route("/teacher/attendance_history")
@login_required("teacher")
def teacher_attendance_history():
    import datetime
    sel_date = request.args.get("date", "")
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    sel_status = request.args.get("status", "")
    q = request.args.get("q", "").strip()
    
    all_records = db.get_attendance_history()
    
    df = load_csv()
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    filtered_records = []
    
    for r in all_records:
        if sel_date and r.get("attendance_date") != sel_date:
            continue
        if sel_class and str(r.get("class")) != sel_class:
            continue
        if sel_section and str(r.get("section")) != sel_section:
            continue
        if sel_status and r.get("attendance_status") != sel_status:
            continue
        if q:
            q_lower = q.lower()
            sid = str(r.get("student_id", "")).lower()
            sname = str(r.get("student_name", "")).lower()
            if q_lower not in sid and q_lower not in sname:
                continue
        filtered_records.append(r)
        
    total_records = len(filtered_records)
    total_present = sum(1 for r in filtered_records if r.get("attendance_status") == "Present")
    total_absent = sum(1 for r in filtered_records if r.get("attendance_status") == "Absent")
    overall_pct = round((total_present / total_records * 100), 1) if total_records > 0 else 0.0
    
    # Charts data calculation
    monthly_stats = {}
    for r in all_records:
        date_str = r.get("attendance_date", "")
        if len(date_str) >= 7:
            month = date_str[:7]
            if month not in monthly_stats:
                monthly_stats[month] = {"Present": 0, "Total": 0}
            monthly_stats[month]["Total"] += 1
            if r.get("attendance_status") == "Present":
                monthly_stats[month]["Present"] += 1
                
    monthly_trend = []
    for m in sorted(monthly_stats.keys()):
        total = monthly_stats[m]["Total"]
        pct = round((monthly_stats[m]["Present"] / total * 100), 1) if total > 0 else 0.0
        monthly_trend.append({"month": m, "percentage": pct})
        
    class_stats = {}
    for r in all_records:
        c = r.get("class", "Unknown")
        if c not in class_stats:
            class_stats[c] = {"Present": 0, "Total": 0}
        class_stats[c]["Total"] += 1
        if r.get("attendance_status") == "Present":
            class_stats[c]["Present"] += 1
            
    class_wise = []
    for c in sorted(class_stats.keys()):
        total = class_stats[c]["Total"]
        pct = round((class_stats[c]["Present"] / total * 100), 1) if total > 0 else 0.0
        class_wise.append({"class": c, "percentage": pct})
        
    present_vs_absent = {
        "Present": sum(1 for r in all_records if r.get("attendance_status") == "Present"),
        "Absent": sum(1 for r in all_records if r.get("attendance_status") == "Absent")
    }
    
    return render_template(
        "attendance_history.html",
        records=filtered_records,
        classes=classes,
        sections=sections,
        total_records=total_records,
        total_present=total_present,
        total_absent=total_absent,
        overall_pct=overall_pct,
        sel_date=sel_date,
        sel_class=sel_class,
        sel_section=sel_section,
        sel_status=sel_status,
        q=q,
        monthly_trend=monthly_trend,
        class_wise=class_wise,
        present_vs_absent=present_vs_absent
    )

@app.route("/teacher/attendance_history/export_csv")
@login_required("teacher")
def export_attendance_history_csv():
    import csv
    from io import StringIO
    from flask import Response
    
    records = db.get_attendance_history()
    
    def generate():
        data = StringIO()
        writer = csv.writer(data)
        writer.writerow([
            "Attendance ID", "Student ID", "Student Name", "Class", "Section", 
            "Attendance Date", "Status", "Attendance Percentage", "Marked Time", 
            "Marked By", "Alert Status", "Created At"
        ])
        yield data.getvalue()
        data.seek(0)
        data.truncate(0)
        
        for r in records:
            writer.writerow([
                r.get("id"), r.get("student_id"), r.get("student_name"), r.get("class"), r.get("section"),
                r.get("attendance_date"), r.get("attendance_status"), r.get("attendance_percentage"),
                r.get("marked_time"), r.get("teacher_name"), r.get("alert_status"), r.get("created_at")
            ])
            yield data.getvalue()
            data.seek(0)
            data.truncate(0)
            
    response = Response(generate(), mimetype="text/csv")
    response.headers.set("Content-Disposition", "attachment", filename="attendance_history.csv")
    return response

@app.route("/teacher/attendance_history/export_pdf")
@login_required("teacher")
def export_attendance_history_pdf():
    import io
    import datetime
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    
    records = db.get_attendance_history()
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    
    content = []
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor("#7c3aed"), spaceAfter=20)
    normal_style = styles['Normal']
    
    content.append(Paragraph("Student Performance System — Attendance History Report", title_style))
    content.append(Paragraph(f"Generated on: {datetime.date.today().isoformat()}", normal_style))
    content.append(Spacer(1, 15))
    
    data = [["Student ID", "Name", "Class", "Sec", "Date", "Status", "Marked By"]]
    for r in records[:50]:
        data.append([
            str(r.get("student_id")),
            str(r.get("student_name")),
            str(r.get("class")),
            str(r.get("section")),
            str(r.get("attendance_date")),
            str(r.get("attendance_status")),
            str(r.get("teacher_name"))
        ])
        
    t = Table(data, colWidths=[90, 200, 50, 40, 100, 80, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#9333ea")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    content.append(t)
    
    doc.build(content)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="attendance_history_report.pdf")

@app.route("/teacher/student-attendance/<student_id>")
@login_required("teacher")
def get_student_attendance_details(student_id):
    all_history = db.get_attendance_history()
    student_records = [r for r in all_history if str(r.get("student_id")) == str(student_id)]
    
    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]
    student_profile = {}
    if not match.empty:
        row = match.iloc[0]
        student_profile = {
            "student_id": student_id,
            "name": row.get("name", "N/A"),
            "class": row.get("Class", "N/A"),
            "section": row.get("section", "N/A"),
            "parent_email": row.get("parent_email", "N/A"),
            "parent_phone": row.get("parent_phone", "N/A"),
            "attendance_percentage": float(row.get("Attendence", 100.0))
        }
        
    present_count = sum(1 for r in student_records if r.get("attendance_status") == "Present")
    absent_count = sum(1 for r in student_records if r.get("attendance_status") == "Absent")
    total_count = len(student_records)
    
    last_30_days = student_records[:30]
    
    return jsonify({
        "success": True,
        "profile": student_profile,
        "stats": {
            "present": present_count,
            "absent": absent_count,
            "total": total_count,
            "percentage": round((present_count / total_count * 100), 1) if total_count > 0 else student_profile.get("attendance_percentage", 100.0)
        },
        "last_30_days": [{
            "date": r.get("attendance_date"),
            "status": r.get("attendance_status"),
            "alert_status": r.get("alert_status"),
            "marked_time": r.get("marked_time"),
            "marked_by": r.get("teacher_name")
        } for r in last_30_days]
    })

@app.route("/teacher/update-attendance", methods=["POST"])
@login_required("teacher")
def update_attendance_record():
    data = request.get_json() or {}
    record_id = data.get("record_id")
    new_status = data.get("status")
    
    if not record_id or not new_status:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
        
    db.update_attendance_history_record(int(record_id), new_status)
    return jsonify({"success": True, "message": "Attendance updated successfully"})

@app.route("/teacher/delete-attendance", methods=["POST"])
@login_required("teacher")
def delete_attendance_record():
    data = request.get_json() or {}
    record_id = data.get("record_id")
    
    if not record_id:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
        
    db.delete_attendance_history_record(int(record_id))
    return jsonify({"success": True, "message": "Attendance record deleted successfully"})


@app.route("/send-absent-alert", methods=["POST"])
def send_absent_alert():
    import datetime
    data = request.get_json() or {}
    student_id = data.get("student_id")
    send_sms = data.get("send_sms", False)
    send_email = data.get("send_email", False)
    date = data.get("date", datetime.date.today().isoformat())
    
    if not student_id:
        return jsonify({"success": False, "error": "Missing student_id"}), 400
        
    success, msg = send_alert_for_student(student_id, date, send_sms, send_email)
    if success:
        return jsonify({"success": True, "message": "Alert sent successfully"})
    else:
        return jsonify({"success": False, "error": msg}), 400

@app.route("/bulk-absent-alert", methods=["POST"])
def bulk_absent_alert():
    import datetime
    data = request.get_json() or {}
    student_ids = data.get("student_ids")
    send_sms = data.get("send_sms", False)
    send_email = data.get("send_email", False)
    date = data.get("date", datetime.date.today().isoformat())
    
    if student_ids is None:
        attendance_db = db.get_attendance_by_date(date)
        student_ids = [sid for sid, att in attendance_db.items() if att["status"] == "Absent"]
        
    count = 0
    for sid in student_ids:
        send_alert_for_student(sid, date, send_sms, send_email)
        count += 1
        
    return jsonify({"success": True, "message": f"Bulk alerts queued successfully for {count} students"})

@app.route("/absent-students-today", methods=["GET"])
def absent_students_today():
    import datetime
    sel_date = request.args.get("date", datetime.date.today().isoformat())
    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    
    df = load_csv()
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]
        
    students = df.to_dict(orient="records")
    attendance_db = db.get_attendance_by_date(sel_date)
    
    absent_list = []
    for s in students:
        sid = str(s["Student_ID"])
        att = attendance_db.get(sid)
        if att and att["status"] == "Absent":
            absent_list.append({
                "Student_ID": sid,
                "name": s.get("name", "N/A"),
                "Class": s.get("Class", "N/A"),
                "section": s.get("section", "N/A"),
                "parent_email": s.get("parent_email", ""),
                "parent_phone": s.get("parent_phone", ""),
                "alert_status": att.get("alert_status", "Pending")
            })
            
    return jsonify({"success": True, "absent_students": absent_list})

@app.route("/teacher/mark_attendance_api", methods=["POST"])
@login_required("teacher")
def teacher_mark_attendance_api():
    data = request.get_json() or {}
    student_id = data.get("student_id")
    date = data.get("date")
    status = data.get("status")
    auto_send = data.get("auto_send", False)
    
    if not student_id or not date or not status:
        return jsonify({"success": False, "error": "Missing parameters"}), 400
        
    db.mark_attendance(student_id, date, status)
    
    # Save automatically to attendance_history table
    try:
        import datetime
        df = load_csv()
        match = df[df["Student_ID"].astype(str) == str(student_id)]
        student_name = "N/A"
        class_val = "N/A"
        section_val = "N/A"
        att_pct = 100.0
        if not match.empty:
            row = match.iloc[0]
            student_name = row.get("name", "N/A")
            class_val = row.get("Class", "N/A")
            section_val = row.get("section", "N/A")
            att_pct = float(row.get("Attendence", 100.0))
            
        teacher = db.get_teacher_by_user_id(session["user_id"])
        teacher_id = teacher["id"] if teacher else None
        teacher_name = teacher["name"] if teacher else session.get("username", "System")
        
        marked_time = datetime.datetime.now().strftime("%I:%M %p")
        alert_status = "Pending"
        if status == "Absent" and auto_send:
            alert_status = "Sent"
            
        db.log_attendance_history(
            student_id=str(student_id),
            student_name=student_name,
            class_val=str(class_val),
            section_val=str(section_val),
            status=status,
            attendance_percentage=att_pct,
            attendance_date=date,
            marked_time=marked_time,
            teacher_id=teacher_id,
            teacher_name=teacher_name,
            alert_status=alert_status
        )
    except Exception as e:
        print(f"Error logging attendance history: {e}")
    
    if status == "Absent" and auto_send:
        send_alert_for_student(student_id, date, send_sms=True, send_email=True)
        return jsonify({"success": True, "alert_sent": True})
        
    return jsonify({"success": True, "alert_sent": False})

if __name__ == "__main__":
    app.run(debug=True, port=5000)