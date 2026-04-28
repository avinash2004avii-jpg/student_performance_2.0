import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")

def migrate_feedback_table():
    print(f"Migrating {DB_PATH} for Recycle Bin support...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. Create a temporary table with the NEW schema
        c.execute("""
        CREATE TABLE pf_new (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id    INTEGER REFERENCES parents(id),
            student_code TEXT    NOT NULL,
            message      TEXT    NOT NULL,
            status       TEXT    DEFAULT 'open' CHECK(status IN ('open','resolved','deleted')),
            created_at   TEXT    DEFAULT (datetime('now')),
            deleted_at   TEXT
        )
        """)

        # 2. Copy data from the old table
        # We need to map existing status and handle missing deleted_at
        c.execute("""
        INSERT INTO pf_new (id, parent_id, student_code, message, status, created_at)
        SELECT id, parent_id, student_code, message, status, created_at FROM parent_feedback
        """)

        # 3. Drop the old table
        c.execute("DROP TABLE parent_feedback")

        # 4. Rename the new table
        c.execute("ALTER TABLE pf_new RENAME TO parent_feedback")

        conn.commit()
        print("Migration successful! Recycle Bin columns and constraints added.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_feedback_table()
