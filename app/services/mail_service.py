import mailer

class MailService:
    @staticmethod
    def send_email_async(to_email, subject, html_content, image_path=None):
        return mailer.send_email_async(to_email, subject, html_content, image_path)

    @staticmethod
    def send_email_sync(to_email, subject, html_content, image_path=None):
        return mailer.send_email_sync(to_email, subject, html_content, image_path)

    @staticmethod
    def get_performance_report_template(student_id, score, attendance, risk, suggestions):
        return mailer.get_performance_report_template(student_id, score, attendance, risk, suggestions)

    @staticmethod
    def get_risk_alert_template(student_name, score, roadmap_link):
        return mailer.get_risk_alert_template(student_name, score, roadmap_link)

    @staticmethod
    def get_feedback_resolved_template(parent_name, student_name):
        return mailer.get_feedback_resolved_template(parent_name, student_name)
