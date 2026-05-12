import os
from flask import Flask, redirect, url_for, session, request
import database as db
from app.config.config import SECRET_KEY

def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = SECRET_KEY

    # Create tables on startup
    db.create_tables()

    # Context processors
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

    # Global redirects / helpers
    @app.route("/")
    def home():
        if "user_id" in session:
            role = session.get("role")
            if role == "admin":   return redirect(url_for("admin_dashboard"))
            if role == "teacher": return redirect(url_for("teacher_dashboard"))
            if role == "student": return redirect(url_for("student_routes.student_dashboard"))
            if role == "parent":  return redirect(url_for("student_routes.parent_dashboard"))
        return redirect(url_for("auth_routes.login_page"))

    # Register blueprints
    from app.routes.auth_routes import auth_blueprint
    from app.routes.student_routes import student_blueprint
    from app.routes.attendance_routes import attendance_blueprint
    from app.routes.analytics_routes import analytics_blueprint
    from app.routes.remarks_routes import remarks_blueprint
    from app.routes.chatbot_routes import chatbot_blueprint

    app.register_blueprint(auth_blueprint)
    app.register_blueprint(student_blueprint)
    app.register_blueprint(attendance_blueprint)
    app.register_blueprint(analytics_blueprint)
    app.register_blueprint(remarks_blueprint)
    app.register_blueprint(chatbot_blueprint)

    return app
