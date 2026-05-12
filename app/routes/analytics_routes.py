import json
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file
import database as db
import pandas as pd
from app.utils.helpers import load_csv
from app.services.prediction_service import PredictionService

analytics_blueprint = Blueprint("analytics_routes", __name__)

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

@analytics_blueprint.route("/teacher/analytics")
@login_required("teacher")
def teacher_analytics():
    df = load_csv()

    sel_class = request.args.get("class", "")
    sel_section = request.args.get("section", "")
    
    classes = sorted(df["Class"].dropna().unique().tolist())
    sections = sorted(df["section"].dropna().unique().tolist())
    
    if sel_class:
        df = df[df["Class"].astype(str) == sel_class]
    if sel_section:
        df = df[df["section"].astype(str) == sel_section]

    df = PredictionService.calculate_display_risk(df)

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
    stable = df[df["risk"] != "At Risk"]

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
        attendance_perf_corr=round(df["Attendence"].corr(df["display_score"]), 2) if total > 1 else 0
    )

import os
from flask import abort
from app.utils.helpers import risk_label, generate_suggestions, create_performance_graph, build_student_report_pdf

@analytics_blueprint.route("/teacher/student-report/<student_id>")
@login_required("teacher")
def student_report(student_id):
    df = load_csv()
    match = df[df["Student_ID"].astype(str) == str(student_id)]

    if match.empty:
        abort(404, description="Student not found")

    row = match.iloc[0].to_dict()

    predicted = PredictionService.predict_score(row)
    risk = risk_label(predicted)

    df_with_display = PredictionService.calculate_display_risk(df.copy())
    class_avg = round(float(df_with_display["display_score"].mean()), 1)
    benchmark_diff = round(predicted - class_avg, 1) if predicted else None
    pct_below = round(float((df_with_display["display_score"] <= predicted).mean() * 100), 1) if predicted else None

    suggestions = generate_suggestions(predicted, row, class_avg=class_avg) if predicted else []

    graph_path = create_performance_graph(student_id, row)

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

    if graph_path and os.path.exists(graph_path):
        try:
            os.remove(graph_path)
        except:
            pass

    return send_file(
        pdf_buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"student_{student_id}_report.pdf"
    )

