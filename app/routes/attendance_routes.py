from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
import database as db
import pandas as pd
from app.utils.helpers import load_csv, risk_label
from app.services.prediction_service import PredictionService
from app.services.mail_service import MailService
from app.services.sms_service import SmsService
from app.models.attendance import AttendanceModel

attendance_blueprint = Blueprint("attendance_routes", __name__)

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

@attendance_blueprint.route("/teacher/absent_alerts")
@login_required("teacher")
def teacher_absent_alerts():
    df = load_csv()
    students_list = df.to_dict(orient="records")
    return render_template("absent_alerts.html", students=students_list)

@attendance_blueprint.route("/teacher/attendance_history")
@login_required("teacher")
def teacher_attendance_history():
    history = AttendanceModel.get_history()
    return render_template("attendance_history.html", history=history)

@attendance_blueprint.route("/teacher/student-attendance/<student_id>")
@login_required("teacher")
def student_attendance_api(student_id):
    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]
    if match.empty:
        return jsonify({"error": "Student not found"}), 404
    student_data = match.iloc[0].to_dict()
    return jsonify({
        "student_id": student_data.get("Student_ID"),
        "name": student_data.get("Student_ID"),
        "class": student_data.get("Class", ""),
        "section": student_data.get("section", ""),
        "attendance": student_data.get("Attendence", 75)
    })

@attendance_blueprint.route("/teacher/mark_attendance_api", methods=["POST"])
@login_required("teacher")
def mark_attendance_api():
    try:
        data = request.json
        student_id = data.get("student_id")
        date = data.get("date")
        status = data.get("status")
        student_name = data.get("student_name")
        class_val = data.get("class_val")
        section_val = data.get("section_val")
        attendance_pct = float(data.get("attendance_pct", 75))
        teacher_name = session.get("username", "Teacher")
        teacher_id = session.get("user_id", 0)

        # Mark in db
        AttendanceModel.mark(student_id, date, status)
        AttendanceModel.log_history(
            student_id, student_name, class_val, section_val, status,
            attendance_pct, date, pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            teacher_id, teacher_name
        )
        return jsonify({"status": "success", "message": "Attendance marked successfully."})
    except Exception as e:
        print("Mark Attendance API Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 400

@attendance_blueprint.route("/send-absent-alert", methods=["POST"])
@login_required("teacher")
def send_absent_alert():
    try:
        data = request.json
        student_id = data.get("student_id")
        student_name = data.get("student_name")
        date = data.get("date")
        
        # Pull parent contact from database
        parent_email = db.get_parent_email_by_student_code(student_id)
        
        message = f"Hello, this is an alert from SPS. Your child (ID {student_id}) has been marked ABSENT on {date}."
        
        if parent_email:
            MailService.send_email_async(
                to_email=parent_email,
                subject=f"⚠️ Attendance Alert: Student {student_id} Absent",
                html_content=f"<html><body><p>{message}</p></body></html>"
            )
        
        # Send SMS too if Twilio is set
        df = load_csv()
        match = df[df["Student_ID"].astype(str) == str(student_id)]
        if not match.empty:
            phone = match.iloc[0].to_dict().get("parent_phone")
            if phone:
                SmsService.send_sms_async(phone, message)
                
        AttendanceModel.update_alert_status(student_id, date, "Sent")
        return jsonify({"status": "success", "message": "Alert sent successfully."})
    except Exception as e:
        print("Send Absent Alert Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 400
