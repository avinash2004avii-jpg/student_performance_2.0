import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database
database.reset_user_password(2, "Roopa123")
print("Successfully reset Roopa's password to: Roopa123")
