"""
train_model.py
Refined training script with data cleaning, feature engineering, and RandomForestRegressor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, os

# File paths
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "students_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("--- Step 1: Loading & Cleaning Data ---")
df = pd.read_csv(DATA_FILE)

# 1. Handle missing values
df["Health_Issues"] = df["Health_Issues"].fillna("None")

# 2. Remove unrealistic values
df = df[df["Sleep_hours"] <= 10]
df = df[df["Attendence"] <= 100]

# 3. Drop unnecessary columns
if "Student_ID" in df.columns:
    df = df.drop(columns=["Student_ID"])

# Optional: drop other non-numeric columns that aren't specified for encoding
# Keeping Class, section, Age, Gender, Parent_Education_Level, etc. for dummies if they exist
# Actually, the user's previous FEATURE_COLS were specific. I'll stick to a defined set.

print("--- Step 2: Feature Engineering ---")
# Required by user
df["total_internal"] = df["internal_test 1"] + df["internal_test 2"]
df["avg_internal"]   = df["total_internal"] / 2

# Additional helpful features (from previous version)
df["study_x_attendance"] = df["Study_hours"] * df["Attendence"] / 100
df["total_score"]        = (df["internal_test 1"] + df["internal_test 2"] + 
                            df["Assignment_score"] + df["Previous_Exam_Score"])

print("--- Step 3: Encoding Categorical Variables ---")
# We'll encode Health_Issues and potentially other categorical columns
categorical_cols = ["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"]
# Only encode if they exist in the dataframe
cols_to_encode = [c for c in categorical_cols if c in df.columns]

df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

print("--- Step 4: Model Training ---")
# Define features and target
TARGET = "Final_Exam_Score"
# We exclude the target and any non-numeric columns that weren't encoded (like Class/section if we don't want them)
# To be safe, let's only take numeric columns as features
X = df_encoded.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors='ignore')
y = df_encoded[TARGET]

# Save column structure for inference alignment
model_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("--- Step 5: Evaluation ---")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"  Accuracy (R2 Score): {r2*100:.2f}%")
print(f"  Mean Absolute Error: {mae:.2f} marks")

print("--- Step 6: Saving Model & Metadata ---")
joblib.dump(model, os.path.join(MODELS_DIR, "student_model.pkl"))
joblib.dump(model_columns, os.path.join(MODELS_DIR, "model_columns.pkl"))

print("Training complete! Model and columns saved.")