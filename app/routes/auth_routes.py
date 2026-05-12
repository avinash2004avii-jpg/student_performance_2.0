from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import database as db
from app.utils.helpers import load_csv

auth_blueprint = Blueprint("auth_routes", __name__)

@auth_blueprint.route("/login", methods=["GET", "POST"])
def login_page():
    return render_template("login.html")

@auth_blueprint.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "admin":
            if user["status"] != "active":
                flash(f"Admin account is {user['status']}. Please check database.", "danger")
                return redirect(url_for("auth_routes.login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("admin_dashboard")) # redirecting to global dashboard
        flash("Invalid credentials or not an admin account.", "danger")
    return render_template("login_admin.html")

@auth_blueprint.route("/login/teacher", methods=["GET", "POST"])
def login_teacher():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "teacher":
            if user["status"] == "pending":
                flash("Your account is pending admin approval.", "warning")
                return redirect(url_for("auth_routes.login_page"))
            if user["status"] == "deactivated":
                flash("Your account has been deactivated.", "danger")
                return redirect(url_for("auth_routes.login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("teacher_dashboard"))
        flash("Invalid credentials or not a teacher account.", "danger")
    return render_template("login_teacher.html")

@auth_blueprint.route("/login/student", methods=["GET", "POST"])
def login_student():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = db.login_user(username, password)
        if user and user["role"] == "student":
            if user["status"] == "deactivated":
                flash("Your account has been deactivated.", "danger")
                return redirect(url_for("auth_routes.login_page"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            return redirect(url_for("student_dashboard"))
        flash("Invalid credentials or not a student account.", "danger")
    return render_template("login_student.html")

@auth_blueprint.route("/login/parent", methods=["GET", "POST"])
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

@auth_blueprint.route("/signup/teacher", methods=["GET", "POST"])
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
                return redirect(url_for("auth_routes.login_page"))
            flash(msg, "danger")
    return render_template("signup_teacher.html")

@auth_blueprint.route("/signup/student", methods=["GET", "POST"])
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
            df = load_csv()
            if student_code not in df["Student_ID"].astype(str).values:
                flash(f"Student ID {student_code} was not found in school records.", "danger")
            else:
                ok, msg = db.signup_student(username, email, password, name,
                                            student_code, class_, section, teacher_id)
                if ok:
                    flash("Account created! Please log in.", "success")
                    return redirect(url_for("auth_routes.login_student"))
                if "student_code" in msg or "UNIQUE" in msg:
                    flash("That Student ID is already registered.", "danger")
                else:
                    flash(msg, "danger")
    return render_template("signup_student.html", teachers=teachers)

@auth_blueprint.route("/signup/parent", methods=["GET", "POST"])
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
                    return redirect(url_for("auth_routes.login_parent"))
                flash(msg, "danger")
    return render_template("signup_parent.html")

@auth_blueprint.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))
