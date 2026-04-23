# 🎓 Student Performance Prediction System

An AI-powered web application that predicts student final exam scores, flags at-risk students, and provides personalised improvement suggestions — built with Flask and a Random Forest model trained on real student data.

---

## Features

- **Three-role system** — Admin, Teacher, Student with separate login pages and dashboards
- **AI Prediction** — ExtraTreesRegressor model with 85%+ accuracy (R²)
- **Single prediction** — Teacher enters student data and gets a predicted score + improvement tips
- **Bulk prediction** — Teacher uploads a CSV of the whole class and gets predictions for every student at once, downloadable as CSV
- **Student dashboard** — Students log in to see their own predicted score, risk level, and personalised improvement tips
- **Admin panel** — Manage all users, view/add/delete students, upload bulk data
- **Signup system** — Teachers and students can register their own accounts

---

## Project Structure

```
student_performance/
├── app.py                  # Flask application — all routes
├── database.py             # SQLite DB setup, auth, user management
├── train_model.py          # Train and save the ML model
├── requirements.txt
│
├── data/
│   ├── students_data.csv   # Main student dataset (900 students)
│   └── sample_bulk.csv     # Sample file for testing bulk predict
│
├── models/                 # Created by train_model.py (git-ignored)
│   ├── student_model.pkl
│   ├── model_columns.pkl
│   └── le_health.pkl
│
└── templates/
    ├── base.html               # Shared layout, all CSS
    ├── home.html               # Landing page
    ├── login.html              # Role selector (fallback)
    ├── login_admin.html
    ├── login_teacher.html
    ├── login_student.html
    ├── signup_teacher.html
    ├── signup_student.html
    ├── admin_dashboard.html
    ├── add_student.html
    ├── teacher_dashboard.html
    ├── teacher_add_student.html
    ├── predict_single.html
    ├── bulk_predict.html
    ├── students_table.html
    └── student_dashboard.html
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance.git
cd student-performance
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML model
```bash
python train_model.py
```
This creates the `models/` folder with the trained model files.

### 5. Run the app
```bash
python app.py
```

Visit `http://localhost:5000`

---

## Default Login

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |

Teachers and students register their own accounts via the signup pages.

---

## How It Works

### ML Model
- **Algorithm**: ExtraTreesRegressor (500 trees)
- **Accuracy**: ~85% R², MAE ~2.6 marks
- **Key features**: Internal test scores, assignment score, previous exam score, study hours, attendance
- **Feature engineering**: 8 derived features including `total_score`, `academic_score`, `study_x_attendance`, `internal_avg`

### Risk Levels
| Score | Level |
|-------|-------|
| < 70 | ⚠ At Risk |
| 70–79 | 📈 Average |
| ≥ 80 | ✅ Safe |

### Student Signup Note
When a student registers, they must enter a **Student ID** that matches the `Student_ID` column in `students_data.csv`. This links their account to their academic record.

---

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn (ExtraTreesRegressor)
- **Database**: SQLite (users/auth), CSV (student data)
- **Frontend**: HTML, CSS (custom dark theme, no frameworks)
- **Templating**: Jinja2
