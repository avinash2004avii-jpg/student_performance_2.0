"""
train_model.py
Run once:  python train_model.py
Saves to:  models/student_model.pkl
           models/model_columns.pkl
           models/le_health.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib, os

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "students_data.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df["Health_Issues"] = df["Health_Issues"].fillna("None")
print(f"  {df.shape[0]} students loaded")

# ── Feature engineering ───────────────────────────────────────────
df["internal_avg"]       = (df["internal_test 1"] + df["internal_test 2"]) / 2
df["internal_diff"]      = abs(df["internal_test 1"] - df["internal_test 2"])
df["academic_score"]     = (df["internal_test 1"] + df["internal_test 2"] + df["Assignment_score"]) / 3
df["study_x_attendance"] = df["Study_hours"] * df["Attendence"] / 100
df["total_score"]        = (df["internal_test 1"] + df["internal_test 2"]
                             + df["Assignment_score"] + df["Previous_Exam_Score"])
df["study_efficiency"]   = df["Study_hours"] / (df["Sleep_hours"] + 1)
df["high_study"]         = (df["Study_hours"] > 4).astype(int)

# ── Only features that matter (>1% importance) ────────────────────
FEATURE_COLS = [
    "Study_hours", "Health_Issues", "Attendence",
    "internal_test 1", "internal_test 2",
    "Assignment_score", "Previous_Exam_Score",
    "internal_avg", "internal_diff", "academic_score",
    "study_x_attendance", "total_score",
    "study_efficiency", "high_study",
]
TARGET = "Final_Exam_Score"

le_health = LabelEncoder()
df["Health_Issues"] = le_health.fit_transform(df["Health_Issues"].astype(str))

X, y = df[FEATURE_COLS], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training ExtraTreesRegressor (500 trees)...")
model = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

r2  = round(model.score(X_test, y_test) * 100, 2)
mae = round(mean_absolute_error(y_test, model.predict(X_test)), 2)
cv  = round(cross_val_score(model, X, y, cv=5, scoring="r2").mean() * 100, 2)
print(f"  Test R²  : {r2}%")
print(f"  CV   R²  : {cv}%")
print(f"  MAE      : {mae} marks avg error")

os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
joblib.dump(model,        os.path.join(MODEL_DIR, "student_model.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "model_columns.pkl"))
joblib.dump(le_health,    os.path.join(MODEL_DIR, "le_health.pkl"))
print("Saved to models/")