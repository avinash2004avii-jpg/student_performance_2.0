import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from app.config.config import DATA_FILE

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

def risk_label(score):
    if score is None: return "Unknown"
    if score < 70:    return "At Risk"
    if score <= 85:   return "Average Performance"
    return "Safe"

def generate_suggestions(score, row, class_avg=None):
    att  = float(row.get("Attendence", row.get("attendance", 100)))
    sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
    it1  = float(row.get("internal_test 1", row.get("internal1", 0)))
    it2  = float(row.get("internal_test 2", row.get("internal2", 0)))
    asgn = float(row.get("Assignment_score", row.get("assignment", 0)))

    if score < 70:
        return [
            ("Attendance", f"Your attendance is quite low right now ({att}%). This can seriously affect your understanding of subjects. Try to attend classes regularly — even improving to 75–85% can make a big difference in your performance."),
            ("Academics (Low Marks)", "Your current scores show that some concepts may not be clear. Start revising basic topics daily and focus on understanding rather than memorizing. Even 1–2 hours of focused study can improve your marks significantly."),
            ("Study Hours", f"Studying {sh} hours is a good start, but increasing it slightly with proper focus can boost your performance. Try creating a simple daily study plan and stick to it."),
            ("Motivation / Overall", "Right now you are in the ‘At Risk’ category, but this is not permanent. With small consistent efforts, you can improve your performance step by step. Start with one subject at a time and build confidence.")
        ]

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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from app.config.config import BASE_DIR

def create_performance_graph(student_id, row):
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
    
    for i, v in enumerate(marks):
        plt.text(i, v + 2, f"{v:.1f}", ha='center', fontweight='bold', fontsize=9)
        
    temp_dir = os.path.join(BASE_DIR, "static", "temp_reports")
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

    content.append(Paragraph("Student Performance Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Student ID:</b> {student_id}", styles["Normal"]))
    content.append(Paragraph(f"<b>Final Exam Score:</b> {row.get('Final_Exam_Score', 'N/A')}", styles["Normal"]))
    content.append(Paragraph(f"<b>Attendance:</b> {row.get('Attendence', 'N/A')}%", styles["Normal"]))
    content.append(Paragraph(f"<b>Study Hours:</b> {row.get('Study_hours', 'N/A')}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Prediction</b>", styles["Heading2"]))
    content.append(Paragraph(f"Predicted Score: {predicted_score}", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))
    content.append(Spacer(1, 12))

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

    if graph_path and os.path.exists(graph_path):
        content.append(Paragraph("<b>Performance Visual Analysis</b>", styles["Heading2"]))
        try:
            img = Image(graph_path, width=400, height=250)
            content.append(img)
        except Exception as e:
            content.append(Paragraph(f"<i>(Graph could not be loaded: {e})</i>", styles["Normal"]))
        content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Suggestions</b>", styles["Heading2"]))

    if suggestions:
        for title, tip in suggestions:
            content.append(Paragraph(f"• {title}: {tip}", styles["Normal"]))
            content.append(Spacer(1, 6))
    else:
        content.append(Paragraph("Great performance! Keep it up.", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

