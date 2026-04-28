import smtplib
import os
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
MAIL_USER   = os.getenv("MAIL_USERNAME", "")   # Your email
MAIL_PASS   = os.getenv("MAIL_PASSWORD", "")   # Your App Password
MAIL_FROM   = os.getenv("MAIL_FROM", "Student Performance System <noreply@school.com>")

def send_email_async(to_email, subject, html_content, image_path=None):
    """Sends email in a separate thread to prevent UI freezing."""
    thread = threading.Thread(target=send_email_sync, args=(to_email, subject, html_content, image_path))
    thread.start()

def send_email_sync(to_email, subject, html_content, image_path=None):
    """Synchronous email sending logic with optional inline image."""
    if not MAIL_USER or not MAIL_PASS:
        print(f"⚠️ MAIL_USERNAME or MAIL_PASSWORD not set.")
        return False

    try:
        # 'related' is essential for inline images
        msg = MIMEMultipart('related')
        msg['From'] = MAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach the HTML content
        msg.attach(MIMEText(html_content, 'html'))

        # Attach the Image (if it exists)
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data)
            # The brackets < > are crucial for the Content-ID header
            image.add_header('Content-ID', '<performance_graph>')
            image.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
            msg.attach(image)
            print(f"✅ Attached inline graph: {image_path}")
        else:
            print(f"⚠️ Graph image not found at: {image_path}")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(MAIL_USER, MAIL_PASS)
            server.send_message(msg)
        
        print(f"✅ Email sent successfully to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False

# ── Email Templates ───────────────────────────────────────────────

def get_performance_report_template(student_id, score, attendance, risk, suggestions):
    suggestion_html = "".join([f"<li style='margin-bottom:8px'><b>{t}:</b> {b}</li>" for t, b in suggestions])
    
    risk_color = "#22c55e" if risk == "Safe" else "#f59e0b" if risk == "Average Performance" else "#ef4444"
    
    return f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b; line-height: 1.6; background-color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="background: #7c3aed; padding: 30px; text-align: center; color: white;">
                <h1 style="margin: 0; font-size: 24px;">Student Performance Report</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.8;">Student ID: {student_id}</p>
            </div>
            <div style="padding: 30px;">
                <p>Dear Parent,</p>
                <p>We are pleased to share the latest academic performance update for your Child.</p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 25px 0;">
                    <div style="background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: bold;">Predicted Score</div>
                        <div style="font-size: 28px; font-weight: 800; color: #7c3aed;">{score}</div>
                    </div>
                    <div style="background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: bold;">Attendance</div>
                        <div style="font-size: 28px; font-weight: 800; color: #0f172a;">{attendance}%</div>
                    </div>
                </div>

                <div style="background: {risk_color}10; border: 1px solid {risk_color}30; padding: 15px; border-radius: 8px; margin-bottom: 25px; text-align: center;">
                    <span style="color: {risk_color}; font-weight: 800; font-size: 18px;">Status: {risk}</span>
                </div>

                <div style="margin: 25px 0; text-align: center;">
                    <h3 style="color: #7c3aed; text-align: left; border-bottom: 2px solid #7c3aed; padding-bottom: 5px;">Performance Analysis</h3>
                    <img src="cid:performance_graph" alt="Performance Graph" style="max-width: 100%; border-radius: 8px; margin-top: 15px; border: 1px solid #e2e8f0;">
                </div>

                <h3 style="color: #7c3aed; border-bottom: 2px solid #7c3aed; padding-bottom: 5px;">Key Improvement Suggestions</h3>
                <ul style="padding-left: 20px;">
                    {suggestion_html}
                </ul>

                <p style="margin-top: 30px;">If you have any questions regarding this report, please feel free to reach out via the Parent Communication Hub.</p>
                
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 30px 0;">
                <p style="font-size: 12px; color: #94a3b8; text-align: center;">
                    This is an automated notification from the <b>Student Performance System</b>.
                </p>
            </div>
        </div>
    </body>
    </html>
    """


def get_risk_alert_template(student_name, score, roadmap_link):
    return f"""
    <html>
    <body style="font-family: sans-serif; color: #333; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; border: 1px solid #eee; border-radius: 10px; overflow: hidden;">
            <div style="background: #7c3aed; padding: 20px; text-align: center; color: white;">
                <h1 style="margin: 0;">Academic Alert 🎓</h1>
            </div>
            <div style="padding: 30px;">
                <p>Hello,</p>
                <p>This is an automated notification regarding <b>{student_name}'s</b> academic performance.</p>
                <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0;">
                    <p style="margin: 0; color: #991b1b; font-weight: bold;">Current Status: AT RISK</p>
                    <p style="margin: 5px 0 0 0;">Predicted Final Score: <span style="font-size: 20px; font-weight: 800;">{score}</span></p>
                </div>
                <p>To help improve these results, our AI has generated a <b>4-Week Intervention Roadmap</b>.</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{roadmap_link}" style="background: #7c3aed; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">View Improvement Plan</a>
                </div>
                <p>Please log in to the Parent Dashboard to view more details and log study hours.</p>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 12px; color: #888; text-align: center;">This is an automated message. Please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """

def get_feedback_resolved_template(parent_name, student_name):
    return f"""
    <html>
    <body style="font-family: sans-serif; color: #333; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; border: 1px solid #eee; border-radius: 10px; overflow: hidden;">
            <div style="background: #22c55e; padding: 20px; text-align: center; color: white;">
                <h1 style="margin: 0;">Concern Resolved ✅</h1>
            </div>
            <div style="padding: 30px;">
                <p>Hello {parent_name},</p>
                <p>The teacher has reviewed and resolved your recent feedback regarding <b>{student_name}</b>.</p>
                <p>If you have further questions, you can submit a new message via the communication hub.</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="#" style="background: #22c55e; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">Go to Dashboard</a>
                </div>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 12px; color: #888; text-align: center;">Student Performance System</p>
            </div>
        </div>
    </body>
    </html>
    """
