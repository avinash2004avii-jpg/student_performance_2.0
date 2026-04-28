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
        role         TEXT    NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
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
        teacher_id   INTEGER REFERENCES teachers(id),
        parent_email TEXT
    );

    CREATE TABLE IF NOT EXISTS parents (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER UNIQUE REFERENCES users(id),
        student_code TEXT    NOT NULL,
        parent_name  TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS parent_feedback (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id    INTEGER REFERENCES parents(id),
        student_code TEXT    NOT NULL,
        message      TEXT    NOT NULL,
        status       TEXT    DEFAULT 'open' CHECK(status IN ('open','resolved','deleted')),
        created_at   TEXT    DEFAULT (datetime('now')),
        deleted_at   TEXT,
        sentiment    TEXT    DEFAULT 'Neutral',
        teacher_reply TEXT,
        is_read_by_parent INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS study_logs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        student_code TEXT    NOT NULL,
        hours        REAL    NOT NULL,
        logged_by    TEXT    NOT NULL,
        log_date     TEXT    DEFAULT (date('now'))
    );
    """)

    # migrate existing db if status column missing
    try:
        c.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'active' CHECK(status IN ('pending','active','deactivated'))")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE parent_feedback ADD COLUMN sentiment TEXT DEFAULT 'Neutral'")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE parent_feedback ADD COLUMN teacher_reply TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE parent_feedback ADD COLUMN is_read_by_parent INTEGER DEFAULT 1")
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
            "INSERT INTO students (user_id,student_code,name,class,section,teacher_id,parent_email) VALUES (?,?,?,?,?,?,?)",
            (user_id, student_code, name, class_, section, teacher_id, None)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()


def signup_parent(username, email, password, parent_name, student_code):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username,email,password,role,status) VALUES (?,?,?,?,?)",
            (username, email, hash_password(password), "parent", "active")
        )
        user_id = conn.execute(
            "SELECT id FROM users WHERE username=?", (username,)
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO parents (user_id, student_code, parent_name) VALUES (?,?,?)",
            (user_id, student_code, parent_name)
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()


# ── Queries ───────────────────────────────────────────────────────
def get_parent_by_user_id(user_id):
    conn = get_conn()
    r = conn.execute(
        "SELECT * FROM parents WHERE user_id=?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(r) if r else None


def log_study_hours(student_code, hours, logged_by):
    conn = get_conn()
    conn.execute(
        "INSERT INTO study_logs (student_code, hours, logged_by) VALUES (?,?,?)",
        (student_code, hours, logged_by)
    )
    conn.commit()
    conn.close()


def submit_feedback(parent_id, student_code, message, sentiment='Neutral'):
    conn = get_conn()
    conn.execute(
        "INSERT INTO parent_feedback (parent_id, student_code, message, sentiment) VALUES (?,?,?,?)",
        (parent_id, student_code, message, sentiment)
    )
    conn.commit()
    conn.close()


def soft_delete_feedback(feedback_id):
    conn = get_conn()
    conn.execute("UPDATE parent_feedback SET status='deleted', deleted_at=datetime('now') WHERE id=?", (feedback_id,))
    conn.commit()
    conn.close()


def restore_feedback(feedback_id):
    conn = get_conn()
    conn.execute("UPDATE parent_feedback SET status='open', deleted_at=NULL WHERE id=?", (feedback_id,))
    conn.commit()
    conn.close()


def purge_deleted_feedback():
    """Permanently delete messages in recycle bin for > 30 days."""
    conn = get_conn()
    conn.execute("DELETE FROM parent_feedback WHERE status='deleted' AND deleted_at < datetime('now', '-30 days')")
    conn.commit()
    conn.close()


def resolve_feedback(feedback_id):
    conn = get_conn()
    conn.execute("UPDATE parent_feedback SET status='resolved' WHERE id=?", (feedback_id,))
    conn.commit()
    conn.close()


def save_teacher_reply(feedback_id, reply_text):
    conn = get_conn()
    conn.execute("""
        UPDATE parent_feedback 
        SET teacher_reply = ?, status = 'resolved', is_read_by_parent = 0 
        WHERE id = ?
    """, (reply_text, feedback_id))
    conn.commit()
    conn.close()


def mark_feedback_as_read(parent_id):
    conn = get_conn()
    conn.execute("UPDATE parent_feedback SET is_read_by_parent = 1 WHERE parent_id = ?", (parent_id,))
    conn.commit()
    conn.close()


def get_unread_reply_count(parent_id):
    conn = get_conn()
    r = conn.execute("SELECT COUNT(*) as count FROM parent_feedback WHERE parent_id = ? AND is_read_by_parent = 0", (parent_id,)).fetchone()
    conn.close()
    return r["count"] if r else 0


def get_open_feedback_count():
    conn = get_conn()
    r = conn.execute("SELECT COUNT(*) as count FROM parent_feedback WHERE status='open'").fetchone()
    conn.close()
    return r["count"] if r else 0


def update_parent_email(student_code, email):
    conn = get_conn()
    conn.execute("UPDATE students SET parent_email = ? WHERE student_code = ?", (email, student_code))
    conn.commit()
    conn.close()


def get_feedbacks_by_parent(parent_id, exclude_deleted=True):
    conn = get_conn()
    if exclude_deleted:
        rows = conn.execute("""
            SELECT * FROM parent_feedback 
            WHERE parent_id = ? AND status != 'deleted'
            ORDER BY created_at DESC
        """, (parent_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM parent_feedback 
            WHERE parent_id = ? AND status = 'deleted'
            ORDER BY created_at DESC
        """, (parent_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_feedback(status=None):
    conn = get_conn()
    query = """
        SELECT pf.*, p.parent_name, s.name as student_name
        FROM parent_feedback pf
        JOIN parents p ON pf.parent_id = p.id
        LEFT JOIN students s ON pf.student_code = s.student_code
    """
    params = []
    if status:
        query += " WHERE pf.status = ?"
        params.append(status)
    
    query += " ORDER BY pf.created_at DESC"
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
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


def get_parent_email_by_student_code(student_code):
    conn = get_conn()
    r = conn.execute("""
        SELECT u.email FROM users u
        JOIN parents p ON u.id = p.user_id
        WHERE p.student_code = ?
    """, (student_code,)).fetchone()
    conn.close()
    return r["email"] if r else None

if __name__ == "__main__":
    create_tables()
    print("DB ready at", DB_PATH)