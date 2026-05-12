import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "data", "students_data.csv")
BULK_OUT = os.path.join(BASE_DIR, "data", "bulk_results.csv")
MDL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "users.db")

SECRET_KEY = os.getenv("SECRET_KEY", "sps_secret_key_change_in_production")

# SMTP Mail config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
MAIL_USER = os.getenv("MAIL_USERNAME", "").strip().strip('"')
MAIL_PASS = os.getenv("MAIL_PASSWORD", "").strip().strip('"')
MAIL_FROM = os.getenv("MAIL_FROM", f"Student Performance System <{MAIL_USER}>")

# Twilio SMS config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
