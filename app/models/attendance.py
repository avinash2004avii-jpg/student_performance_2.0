import database as db

class AttendanceModel:
    @staticmethod
    def get_by_date(date):
        return db.get_attendance_by_date(date)

    @staticmethod
    def mark(student_id, date, status):
        return db.mark_attendance(student_id, date, status)

    @staticmethod
    def update_alert_status(student_id, date, alert_status):
        return db.update_attendance_alert_status(student_id, date, alert_status)

    @staticmethod
    def log_history(student_id, student_name, class_val, section_val, status, pct, date, marked_time, teacher_id, teacher_name):
        return db.log_attendance_history(student_id, student_name, class_val, section_val, status, pct, date, marked_time, teacher_id, teacher_name)

    @staticmethod
    def get_history():
        return db.get_attendance_history()

    @staticmethod
    def update_history_record(record_id, status, alert_status=None):
        return db.update_attendance_history_record(record_id, status, alert_status)

    @staticmethod
    def delete_history_record(record_id):
        return db.delete_attendance_history_record(record_id)
