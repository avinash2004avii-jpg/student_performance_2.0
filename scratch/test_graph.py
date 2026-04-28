import os, sys
# Add current dir to path
sys.path.append(os.getcwd())

from app import create_performance_graph

test_row = {
    'internal_test 1': 70,
    'internal_test 2': 75,
    'Assignment_score': 80,
    'Previous_Exam_Score': 72,
    'Final_Exam_Score': 78
}

path = create_performance_graph('TEST101', test_row)
print(f"Graph saved to: {path}")
print(f"File exists: {os.path.exists(path)}")
