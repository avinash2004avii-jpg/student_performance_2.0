import mailer

class SmsService:
    @staticmethod
    def send_sms_async(to_phone, message):
        return mailer.send_sms_async(to_phone, message)

    @staticmethod
    def send_sms_sync(to_phone, message):
        return mailer.send_sms_sync(to_phone, message)
