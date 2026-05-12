import database as db

class AlertsModel:
    @staticmethod
    def get_unread_reply_count(parent_id):
        return db.get_unread_reply_count(parent_id)

    @staticmethod
    def get_open_feedback_count():
        return db.get_open_feedback_count()
