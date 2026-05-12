import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=== ALL USERS ===")
rows = conn.execute("SELECT id, username, email, role, status FROM users").fetchall()
for r in rows:
    print(dict(r))

print("\n=== TEACHERS ===")
rows = conn.execute("SELECT * FROM teachers").fetchall()
for r in rows:
    print(dict(r))

conn.close()
