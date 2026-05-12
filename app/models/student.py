import database as db

class StudentModel:
    @staticmethod
    def get_by_user_id(user_id):
        return db.get_student_by_user_id(user_id)

    @staticmethod
    def signup(username, email, password, name, student_code, class_=None, section=None, teacher_id=None):
        return db.signup_student(username, email, password, name, student_code, class_, section, teacher_id)

    @staticmethod
    def update_parent_email(student_code, email):
        return db.update_parent_email(student_code, email)

    @staticmethod
    def get_parent_email(student_code):
        return db.get_parent_email_by_student_code(student_code)
