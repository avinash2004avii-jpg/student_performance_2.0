import sqlite3
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "users.db")
DATA_FILE = os.path.join(BASE, "data", "students_data.csv")

# 1. Update SQLite
try:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("ALTER TABLE students ADD COLUMN parent_email TEXT")
    conn.commit()
    conn.close()
    print("SQLite students table updated.")
except Exception as e:
    print(f"SQLite update info: {e}")

# 2. Update CSV
try:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "parent_email" not in df.columns:
            df["parent_email"] = ""
            df.to_csv(DATA_FILE, index=False)
            print("CSV updated with parent_email column.")
        else:
            print("CSV already has parent_email column.")
    else:
        print("CSV file not found at " + DATA_FILE)
except Exception as e:
    print(f"CSV update error: {e}")
