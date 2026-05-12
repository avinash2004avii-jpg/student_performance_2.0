import os
import joblib
import pandas as pd
import numpy as np
from app.config.config import MDL_DIR

class PredictionService:
    _model = None
    _model_columns = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            p_model = os.path.join(MDL_DIR, "student_model.pkl")
            p_cols = os.path.join(MDL_DIR, "model_columns.pkl")
            if os.path.exists(p_model) and os.path.exists(p_cols):
                try:
                    cls._model = joblib.load(p_model)
                    cls._model_columns = joblib.load(p_cols)
                    print("✅ ML Model loaded successfully.")
                except Exception as e:
                    print(f"❌ Error loading ML model: {e}")
            else:
                print(f"⚠️ ML Model files not found in {MDL_DIR}.")
        return cls._model, cls._model_columns

    @classmethod
    def build_features(cls, row):
        model, model_columns = cls.load_model()
        if model is None:
            return None

        it1  = float(row.get("internal_test 1",  row.get("internal1",  0)))
        it2  = float(row.get("internal_test 2",  row.get("internal2",  0)))
        asgn = float(row.get("Assignment_score", row.get("assignment", 0)))
        prev = float(row.get("Previous_Exam_Score", row.get("previous", 0)))
        att  = float(row.get("Attendence",  row.get("attendance",  75)))
        sh   = float(row.get("Study_hours", row.get("study_hours", 3)))
        slp  = float(row.get("Sleep_hours", row.get("sleep_hours", 7)))

        total_internal = it1 + it2
        avg_internal   = total_internal / 2

        data = {
            "Study_hours": sh, "Sleep_hours": slp, "Attendence": att,
            "internal_test 1": it1, "internal_test 2": it2,
            "Assignment_score": asgn, "Previous_Exam_Score": prev,
            "total_internal": total_internal,
            "avg_internal": avg_internal,
            "study_x_attendance": sh * att / 100,
            "total_score": it1 + it2 + asgn + prev,
            "study_efficiency": sh / (slp + 1),
            "high_study": 1 if sh > 4 else 0,
        }

        hlth = row.get("Health_Issues", row.get("health", "None"))
        gender = row.get("Gender", "Male")
        internet = row.get("Internet_Access", "Yes")
        extra = row.get("Extracurricular_Activities", "No")

        data["Health_Issues"] = hlth
        data["Gender"] = gender
        data["Internet_Access"] = internet
        data["Extracurricular_Activities"] = extra

        df_row = pd.DataFrame([data])
        cols_to_encode = ["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"]
        df_encoded = pd.get_dummies(df_row, columns=cols_to_encode, drop_first=True)

        final_features = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in model_columns:
            if col in df_encoded.columns:
                final_features[col] = df_encoded[col].values
            elif col in data:
                final_features[col] = data[col]

        return final_features

    @classmethod
    def predict_score(cls, row):
        model, _ = cls.load_model()
        if model is None:
            return None
        try:
            features = cls.build_features(row)
            return round(float(model.predict(features)[0]), 1)
        except Exception as e:
            print(f"❌ Prediction Error: {e}")
            return None

    @classmethod
    def calculate_display_risk(cls, df):
        if df.empty:
            df["display_score"] = []
            df["risk"] = []
            df["risk_val"] = []
            return df
        
        df["display_score"] = df["Final_Exam_Score"].astype(float)
        mask = df["display_score"] <= 0
        
        model, model_columns = cls.load_model()
        if mask.any() and model is not None:
            sub = df[mask].copy()
            it1 = sub.get("internal_test 1", sub.get("internal1", pd.Series(0, index=sub.index))).astype(float)
            it2 = sub.get("internal_test 2", sub.get("internal2", pd.Series(0, index=sub.index))).astype(float)
            asgn = sub.get("Assignment_score", sub.get("assignment", pd.Series(0, index=sub.index))).astype(float)
            prev = sub.get("Previous_Exam_Score", sub.get("previous", pd.Series(0, index=sub.index))).astype(float)
            att = sub.get("Attendence", sub.get("attendance", pd.Series(75, index=sub.index))).astype(float)
            sh = sub.get("Study_hours", sub.get("study_hours", pd.Series(3, index=sub.index))).astype(float)
            slp = sub.get("Sleep_hours", sub.get("sleep_hours", pd.Series(7, index=sub.index))).astype(float)

            X_data = {
                "Study_hours": sh, "Sleep_hours": slp, "Attendence": att,
                "internal_test 1": it1, "internal_test 2": it2,
                "Assignment_score": asgn, "Previous_Exam_Score": prev,
                "total_internal": it1 + it2,
                "avg_internal": (it1 + it2) / 2,
                "study_x_attendance": sh * att / 100,
                "total_score": it1 + it2 + asgn + prev,
                "study_efficiency": sh / (slp + 1),
                "high_study": (sh > 4).astype(int),
            }
            
            for col in ["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"]:
                X_data[col] = sub.get(col, pd.Series("None" if col=="Health_Issues" else "Yes", index=sub.index))
                
            X_df = pd.DataFrame(X_data)
            X_encoded = pd.get_dummies(X_df, columns=["Health_Issues", "Gender", "Internet_Access", "Extracurricular_Activities"], drop_first=True)
            
            final_X = pd.DataFrame(0, index=sub.index, columns=model_columns)
            for col in model_columns:
                if col in X_encoded.columns:
                    final_X[col] = X_encoded[col]
                elif col in X_df.columns:
                    final_X[col] = X_df[col]
            
            try:
                preds = model.predict(final_X)
                df.loc[mask, "display_score"] = np.round(preds, 1)
            except Exception as e:
                print(f"Batch Prediction Error: {e}")

        df["risk"] = np.where(df["display_score"] < 70, "At Risk",
                     np.where(df["display_score"] <= 85, "Average Performance", "Safe"))
        df["risk_val"] = np.where(df["display_score"] < 70, "1", "0")
        
        return df

    @classmethod
    def calculate_sensitivity(cls, row, current_score):
        if current_score >= 75:
            return None
        
        results = []
        
        # 1. Try Attendance
        test_row = row.copy()
        orig_att = float(row.get('Attendence', 75))
        if orig_att < 95:
            needed_att = min(100.0, orig_att + 10.0)
            test_row['Attendence'] = needed_att
            new_score = cls.predict_score(test_row)
            if new_score and new_score > current_score:
                results.append({"metric": "Attendance", "from": orig_att, "to": needed_att, "gain": round(new_score - current_score, 1)})

        # 2. Try Study Hours
        test_row = row.copy()
        orig_sh = float(row.get('Study_hours', 3))
        if orig_sh < 8:
            needed_sh = orig_sh + 2.0
            test_row['Study_hours'] = needed_sh
            new_score = cls.predict_score(test_row)
            if new_score and new_score > current_score:
                results.append({"metric": "Study Hours", "from": orig_sh, "to": needed_sh, "gain": round(new_score - current_score, 1)})
                
        # 3. Try Next Internal
        test_row = row.copy()
        orig_it2 = float(row.get('internal_test 2', 0))
        if orig_it2 < 90:
            needed_it2 = min(100.0, orig_it2 + 15.0)
            test_row['internal_test 2'] = needed_it2
            new_score = cls.predict_score(test_row)
            if new_score and new_score > current_score:
                results.append({"metric": "Next Internal Test", "from": orig_it2, "to": needed_it2, "gain": round(new_score - current_score, 1)})

        return results
