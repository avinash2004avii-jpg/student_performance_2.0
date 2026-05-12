import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

with open(APP_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_avg.txt")
with open(out_path, "w", encoding="utf-8") as out:
    for i, line in enumerate(lines):
        if "avg" in line.lower() or "mean" in line.lower():
            out.write(f"Line {i+1}: {line.strip()}\n")

print("Successfully wrote search results to search_avg.txt")
