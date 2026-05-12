import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

keywords = ["class_average", "0.0", "Benchmark", "Class Average", "report", "download", "pdf"]

with open(APP_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for kw in keywords:
    print(f"=== SEARCHING FOR: {kw} ===")
    count = 0
    for i, line in enumerate(lines):
        if kw.lower() in line.lower():
            print(f"Line {i+1}: {line.strip()}")
            count += 1
            if count >= 30:
                print("... truncated ...")
                break
