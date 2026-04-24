"""
database.py  —  single source of truth for all DB operations
Schema:
  users        (id, username, email, password_hash, role, created_at)
  teachers     (id, user_id, name, subject)
  students     (id, user_id, student_code, name, class, section, teacher_id)
"""

import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_tables():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT    UNIQUE NOT NULL,
        email        TEXT    UNIQUE NOT NULL,
        password     TEXT    NOT NULL,
        role         TEXT    NOT NULL CHECK(role IN ('admin','teacher','student')),
        status       TEXT    DEFAULT 'pending' CHECK(status IN ('pending','active','deactivated')),
        created_at   TEXT    DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS teachers (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER UNIQUE REFERENCES users(id),
        name         TEXT    NOT NULL,
        subject      TEXT    DEFAULT 'General'
    );

    CREATE TABLE IF NOT EXISTS students (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER UNIQUE REFERENCES users(id),
        student_code TEXT    UNIQUE NOT NULL,
        name         TEXT    NOT NULL,
        class        TEXT,
        section      TEXT,
        teacher_id   INTEGER REFERENCES teachers(id)
    );
    """)

    # migrate existing db if status column missing
    try:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active' CHECK(status IN ('pending','active','deactivated'))")
    except sqlite3.OperationalError:
        pass

    # seed admin if not exists
    admin_exists = c.execute(
        "SELECT 1 FROM users WHERE role='admin'"
    ).fetchone()

    if not admin_exists:
        c.execute(
            "INSERT INTO users (username,email,password,role,status) VALUES (?,?,?,?,?)",
            ("admin", "admin@school.com", hash_password("admin123"), "admin", "active")
        )
        print("✅ Default admin created  →  username: admin  password: admin123")

    conn.commit()
    conn.close()


# ── Auth ──────────────────────────────────────────────────────────
def login_user(username: str, password: str):
    """Returns user Row or None."""
    conn = get_conn()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    ).fetchone()
    conn.close()
    return user


def username_exists(username: str) -> bool:
    conn = get_conn()
    r = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return r is not None


def email_exists(email: str) -> bool:
    conn = get_conn()
    r = conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    return r is not None


# ── Signup ────────────────────────────────────────────────────────
def signup_teacher(username, email, password, name, subject="General"):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username,email,password,role,status) VALUES (?,?,?,?,?)",
            (username, email, hash_password(password), "teacher", "pending")
        )
        user_id = conn.execute(
            "SELECT id FROM users WHERE username=?", (username,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO teachers (user_id,name,subject) VALUES (?,?,?)",
            (user_id, name, subject)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()


def signup_student(username, email, password, name, student_code,
                   class_=None, section=None, teacher_id=None):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username,email,password,role,status) VALUES (?,?,?,?,?)",
            (username, email, hash_password(password), "student", "active")
        )
        user_id = conn.execute(
            "SELECT id FROM users WHERE username=?", (username,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO students (user_id,student_code,name,class,section,teacher_id) VALUES (?,?,?,?,?,?)",
            (user_id, student_code, name, class_, section, teacher_id)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()


# ── Queries ───────────────────────────────────────────────────────
def get_all_teachers():
    conn = get_conn()
    rows = conn.execute("""
        SELECT t.id, t.user_id, t.name, t.subject, u.username, u.email, u.created_at
        FROM teachers t JOIN users u ON t.user_id=u.id
        ORDER BY t.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]



def get_teacher_by_user_id(user_id):
    conn = get_conn()
    r = conn.execute(
        "SELECT * FROM teachers WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(r) if r else None


def get_student_by_user_id(user_id):
    conn = get_conn()
    r = conn.execute(
        "SELECT * FROM students WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(r) if r else None


def get_all_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id,username,email,role,status,created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_user_status(user_id: int, status: str):
    conn = get_conn()
    conn.execute("UPDATE users SET status=? WHERE id=?", (status, user_id))
    conn.commit()
    conn.close()

def reset_user_password(user_id: int, new_password: str):
    conn = get_conn()
    conn.execute("UPDATE users SET password=? WHERE id=?", 
                 (hash_password(new_password), user_id))
    conn.commit()
    conn.close()

def delete_user(user_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def get_all_teachers_simple():
    """For dropdowns: returns list of (id, name)."""
    conn = get_conn()
    rows = conn.execute("SELECT id, name FROM teachers ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    create_tables()
    print("DB ready at", DB_PATH)