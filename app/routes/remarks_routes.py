from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import database as db
from app.utils.helpers import load_csv

remarks_blueprint = Blueprint("remarks_routes", __name__)

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

import csv
import io
from flask import send_file

@remarks_blueprint.route("/teacher/remarks", methods=["GET"])
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
        
    # Sort students by Student_ID in ascending order
    if not df.empty:
        try:
            df = df.iloc[pd.to_numeric(df["Student_ID"], errors='coerce').argsort()]
        except Exception:
            df = df.sort_values(by="Student_ID")
            
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

@remarks_blueprint.route("/teacher/remarks/add", methods=["POST"])
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
        
    return redirect(url_for("remarks_routes.teacher_remarks", student_id=student_id, q=q, section=sel_section, **{"class": sel_class}))

@remarks_blueprint.route("/teacher/remarks/delete", methods=["POST"])
@login_required("teacher")
def delete_remark():
    remark_id = request.form.get("remark_id")
    student_id = request.form.get("student_id")
    q = request.form.get("q", "")
    sel_class = request.form.get("class", "")
    sel_section = request.form.get("section", "")
    
    if remark_id:
        db.delete_teacher_remark(remark_id)
        flash("Remark deleted successfully.", "warning")
        
    return redirect(url_for("remarks_routes.teacher_remarks", student_id=student_id, q=q, section=sel_section, **{"class": sel_class}))

@remarks_blueprint.route("/teacher/remarks/download-csv")
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


@remarks_blueprint.route("/teacher/feedback")
@login_required("teacher")
def teacher_feedback():
    feedbacks = db.get_all_feedback()
    return render_template("teacher_feedback.html", feedback_list=feedbacks)

@remarks_blueprint.route("/teacher/reply-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def reply_feedback(fid):
    reply_text = request.form.get("reply", "").strip()
    if reply_text:
        db.save_teacher_reply(fid, reply_text)
        flash("Reply saved and concern resolved.", "success")
    return redirect(url_for("remarks_routes.teacher_feedback"))

@remarks_blueprint.route("/teacher/resolve-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def resolve_feedback(fid):
    db.resolve_feedback(fid)
    flash("Concern marked as resolved.", "success")
    return redirect(url_for("remarks_routes.teacher_feedback"))

@remarks_blueprint.route("/teacher/delete-feedback/<int:fid>", methods=["POST"])
@login_required("teacher")
def delete_feedback(fid):
    db.soft_delete_feedback(fid)
    flash("Feedback moved to recycle bin.", "warning")
    return redirect(url_for("remarks_routes.teacher_feedback"))
