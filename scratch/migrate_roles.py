import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "users.db")

def migrate_users_table():
    print(f"Migrating {DB_PATH} to support 'parent' role...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        # 1. Create a temporary table with the NEW schema
        c.execute("""
        CREATE TABLE users_new (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    UNIQUE NOT NULL,
            email        TEXT    UNIQUE NOT NULL,
            password     TEXT    NOT NULL,
            role         TEXT    NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
            status       TEXT    DEFAULT 'pending' CHECK(status IN ('pending','active','deactivated')),
            created_at   TEXT    DEFAULT (datetime('now'))
        )
        """)

        # 2. Copy data from the old table
        # We assume the columns match the order or specify them
        c.execute("""
        INSERT INTO users_new (id, username, email, password, role, status, created_at)
        SELECT id, username, email, password, role, status, created_at FROM users
        """)

        # 3. Drop the old table
        c.execute("DROP TABLE users")

        # 4. Rename the new table to 'users'
        c.execute("ALTER TABLE users_new RENAME TO users")

        conn.commit()
        print("Migration successful! The 'parent' role is now supported.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_users_table()
