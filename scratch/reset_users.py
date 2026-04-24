import database as db
conn = db.get_conn()
# Reset all teachers and students to pending so the user can test the approval system
conn.execute("UPDATE users SET status='pending' WHERE role IN ('teacher', 'student') AND username != 'admin'")
conn.commit()
conn.close()
print("Success: All non-admin users reset to pending.")
