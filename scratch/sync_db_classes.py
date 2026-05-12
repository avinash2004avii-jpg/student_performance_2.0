import sqlite3
import pandas as pd
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE, "data", "students_data.csv")
DB_PATH = os.path.join(BASE, "users.db")

def sync_database_classes():
    print(f"Loading updated CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Connecting to database {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    updated_count = 0
    for idx, row in df.iterrows():
        sid = str(row["Student_ID"])
        cls = str(row["Class"])
        sec = str(row["section"])
        
        cursor.execute(
            "UPDATE students SET class = ?, section = ? WHERE student_code = ?",
            (cls, sec, sid)
        )
        if cursor.rowcount > 0:
            updated_count += cursor.rowcount
            
    conn.commit()
    conn.close()
    print(f"Successfully synchronized {updated_count} students in the SQLite database to match the new ID ranges!")

if __name__ == "__main__":
    sync_database_classes()
