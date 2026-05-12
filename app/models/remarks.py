import database as db

class RemarksModel:
    @staticmethod
    def add(student_code, teacher_id, remark):
        return db.add_teacher_remark(student_code, teacher_id, remark)

    @staticmethod
    def get_by_student(student_code):
        return db.get_remarks_by_student(student_code)

    @staticmethod
    def delete(remark_id):
        return db.delete_teacher_remark(remark_id)
