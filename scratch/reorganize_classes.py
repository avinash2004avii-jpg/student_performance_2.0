import pandas as pd
import os

CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "students_data.csv"))

def reorganize_students():
    print(f"Loading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    classes = []
    sections = []
    
    for idx, row in df.iterrows():
        try:
            sid = int(row["Student_ID"])
        except (ValueError, KeyError):
            classes.append(row.get("Class"))
            sections.append(row.get("section"))
            continue
            
        if 1 <= sid <= 100:
            classes.append("8th")
            sections.append("A")
        elif 101 <= sid <= 200:
            classes.append("8th")
            sections.append("B")
        elif 201 <= sid <= 300:
            classes.append("8th")
            sections.append("C")
        elif 301 <= sid <= 400:
            classes.append("9th")
            sections.append("A")
        elif 401 <= sid <= 500:
            classes.append("9th")
            sections.append("B")
        elif 501 <= sid <= 600:
            classes.append("9th")
            sections.append("C")
        elif 601 <= sid <= 700:
            classes.append("10th")
            sections.append("A")
        elif 701 <= sid <= 800:
            classes.append("10th")
            sections.append("B")
        elif 801 <= sid <= 910: # covering up to 910 just in case
            classes.append("10th")
            sections.append("C")
        else:
            classes.append("10th")
            sections.append("C")
            
    df["Class"] = classes
    df["section"] = sections
    
    df.to_csv(CSV_PATH, index=False)
    print("Successfully reorganized students based on ID ranges!")

if __name__ == "__main__":
    reorganize_students()
