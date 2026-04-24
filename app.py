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
matplotlib.use('Agg') # Necessary for non-GUI environments
import matplotlib.pyplot as plt

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

# ── Helpers ───────────────────────────────────────────────────────
def load_csv():
    df = pd.read_csv(DATA_FILE)
    df["Health_Issues"] = df["Health_Issues"].fillna("None")
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
    if score < 85:    return "Average Performance"
    return "Safe"


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
    df_encoded = pd.get_dummies(df_row, columns=cols_to_encode)

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
    
    if slp < 6:
        tips.append(("Sleep", f"Only {slp} hours of sleep affects memory and focus. 7–8 hours of sleep helps retain what you study."))
    
    if it1 < 50 and it2 < 50:
        tips.append(("Core Concepts", "Both internal tests are below 50. Focus on understanding fundamentals rather than memorising — consider extra tutoring."))
    
    if not tips:
        tips.append(("Achievement", "Keep up the great work and maintain consistency!"))

    return tips

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
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", 
                          np.where(df["Final_Exam_Score"] < 85, "Average Performance", "Safe"))
    
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
        df = df[df["Final_Exam_Score"] < 70]

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
        
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", 
                          np.where(df["Final_Exam_Score"] < 85, "Average Performance", "Safe"))
    
    total = len(df)
    at_risk_count = len(df[df["Final_Exam_Score"] < 70])
    avg_score = round(df["Final_Exam_Score"].mean(), 1) if not df.empty else 0
    avg_att = round(df["Attendence"].mean(), 1) if not df.empty else 0
    risk_pct = round((at_risk_count / total * 100), 1) if total > 0 else 0
    
    # Limit to 10 for dashboard preview
    risk_students = df[df["risk"] == "At Risk"].head(10).to_dict(orient="records")
    
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
        sel_section=sel_section
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
    
    if request.method == "POST":
        sid = request.form.get("student_id", "").strip()
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
            except Exception as e:
                flash(f"Prediction error: {e}", "danger")
        else:
            if sid:
                flash(f"Student ID '{sid}' not found in our records.", "warning")
                
    return render_template("auto_predict.html",
                           result=result, score=score, 
                           suggestions=suggestions, student=student)

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
        except Exception:
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
        results=results, show_results=True,
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
    df["risk_val"] = np.where(df["Final_Exam_Score"] < 70, "1", "0")
    df["risk"] = np.where(df["Final_Exam_Score"] < 70, "At Risk", 
                          np.where(df["Final_Exam_Score"] < 85, "Average Performance", "Safe"))
    
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

@app.route("/teacher/upload-students", methods=["POST"])
@login_required("teacher")
def teacher_upload_students():
    file = request.files.get("file")
    if not file:
        flash("No file uploaded.", "danger")
        return redirect(url_for("teacher_students"))
    try:
        new_df = pd.read_csv(file) if file.filename.endswith(".csv") else pd.read_excel(file)
        new_df["Health_Issues"] = new_df["Health_Issues"].fillna("None")
    except Exception as e:
        flash(f"Could not read file: {e}", "danger")
        return redirect(url_for("teacher_students"))
    existing = load_csv()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Student_ID"], keep="last")
    combined.to_csv(DATA_FILE, index=False)
    flash(f"Uploaded {len(new_df)} student(s).", "success")
    return redirect(url_for("teacher_students"))

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
        
        # Required columns mapping (CSV headers to project column names)
        # Using a dictionary to normalize headers if they are slightly different
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
            'health_issues': 'Health_Issues'
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
        
        # Avoid duplicate student IDs - keep newest
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Student_ID"], keep="last")
        
        df_combined.to_csv(DATA_FILE, index=False)
        flash(f"Successfully processed {len(df_new)} students from CSV.", "success")
        
    except Exception as e:
        flash(f"Error processing CSV: {e}", "danger")
        
    return redirect(url_for("teacher_students"))

@app.route("/teacher/sample_csv")
@login_required("teacher")
def teacher_sample_csv():
    headers = "student_id,class,section,attendance,internal_test_1,internal_test_2,assignment_score,previous_exam_score,study_hours,sleep_hours,health_issues\n"
    sample_row = "S9999,10th,A,85.5,70,75,80,72,4,7,None"
    output = io.BytesIO()
    output.write((headers + sample_row).encode('utf-8'))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="sample_students.csv")


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

    if student:
        match = df[df["Student_ID"].astype(str) == str(student["student_code"])]
        if not match.empty:
            row   = match.iloc[0].to_dict()
            score = predict_score(row)
            result = risk_label(score)
            if score is not None and score < 80:
                suggestions = generate_suggestions(score, row)

    return render_template("student_dashboard.html",
        student=student, row=row, score=score,
        result=result, suggestions=suggestions,
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

    # Basic analytics data (safe defaults)
    hist_labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
    hist_counts = [0] * 10

    for score in df["Final_Exam_Score"]:
        idx = min(int(score // 10), 9)
        hist_counts[idx] += 1

    scatter_points = [
        {"x": row["Attendence"], "y": row["Final_Exam_Score"]}
        for _, row in df.iterrows()
    ]

    component_labels = ["Internal 1","Internal 2","Assignment", "Previous"]

    at_risk = df[df["Final_Exam_Score"] < 70]
    safe = df[df["Final_Exam_Score"] >= 70]

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
    avg_score = round(df["Final_Exam_Score"].mean(), 1) if total > 0 else 0
    avg_att = round(df["Attendence"].mean(), 1) if total > 0 else 0
    class_avg = df.groupby("Class")["Final_Exam_Score"].mean().round(1).to_dict() if total > 0 else {}

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
        attendance_perf_corr=round(df["Attendence"].corr(df["Final_Exam_Score"]),2) if total > 1 else 0
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

        # ✅ Get student from DB
        student = db.get_student_by_user_id(session["user_id"])

        if not student:
            return jsonify({"reply": "❌ Student not found in database."})

        student_id = str(student.get("student_code", "")).strip().upper()

        if not student_id:
            return jsonify({"reply": "⚠️ Student ID missing."})

        # ✅ Load CSV
        try:
            df = load_csv()
        except Exception as e:
            return jsonify({"reply": f"⚠️ Error loading data file: {e}"})

        # ✅ Normalize CSV IDs
        df["Student_ID"] = df["Student_ID"].astype(str).str.strip().str.upper()

        # ✅ Match student
        match = df[df["Student_ID"] == student_id]

        if match.empty:
            return jsonify({"reply": f"⚠️ No data found for Student ID: {student_id}"})

        # ✅ Extract row
        row = match.iloc[0].to_dict()

        # ✅ Predict
        score = predict_score(row)
        risk = risk_label(score)
        suggestions = generate_suggestions(score if score else 0, row)

        # ✅ Build prompt
        prompt = f"""
You are an academic assistant.

Student Details:
- Score: {score}
- Risk: {risk}
- Attendance: {row.get('Attendence')}
- Study Hours: {row.get('Study_hours')}
- Internal 1: {row.get('internal_test 1')}
- Internal 2: {row.get('internal_test 2')}
- Assignment: {row.get('Assignment_score')}

Suggestions:
{', '.join([tip[1] for tip in suggestions])}

User Question: {user_message}

Give clear, short, helpful answer.
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

        except Exception as e:
            print("Gemini Error:", e)

            # ✅ SMART FALLBACK
            if "score" in user_message:
                reply = f"📊 Your predicted score is {score} ({risk})"

            elif "why" in user_message or "low" in user_message:
                reasons = explain_prediction(row)
                reply = "📌 Reasons:\n" + "\n".join(reasons)

            elif "improve" in user_message or "suggest" in user_message:
                if suggestions:
                    reply = "💡 Suggestions:\n" + "\n".join([tip[1] for tip in suggestions])
                else:
                    reply = "👍 Your performance is already good."

            else:
                reply = "⚠️ AI unavailable. Showing basic analysis.\n"
                reply += f"Score: {score} ({risk})"

        return jsonify({"reply": reply})

    except Exception as e:
        print("Chatbot Error:", e)
        return jsonify({"reply": "❌ Something went wrong. Please try again."})
if __name__ == "__main__":
    app.run(debug=True, port=5000)