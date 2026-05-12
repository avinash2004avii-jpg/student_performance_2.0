import os
import sys
import random
import sqlite3
import datetime
import pandas as pd

# Set up path to import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database as db

# Load students CSV data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "students_data.csv")

def run_migration_and_generation():
    print("Starting Remarks Database Migration & Sample Generation...")
    
    # Repairing existing records with N/A values
    print("Repairing existing records with N/A values...")
    conn = db.get_conn()
    df_students = pd.read_csv(DATA_FILE)
    
    # Create student details lookup dict
    students_lookup = {}
    for idx, row in df_students.iterrows():
        sid = str(row["Student_ID"]).strip()
        students_lookup[sid] = {
            "name": row.get("name", f"Student {sid}"),
            "class": str(row.get("Class", "10")),
            "section": str(row.get("section", "A"))
        }
        
    teachers_list = [
        {"id": 1, "name": "Roopa", "subject": "Science"},
        {"id": 2, "name": "nishu", "subject": "english"},
        {"id": 3, "name": "bhagya", "subject": "mathematics"},
        {"id": 4, "name": "Test Teacher", "subject": "CS"},
        {"id": 5, "name": "Nishu Teacher", "subject": "Computer Science"},
        {"id": 6, "name": "Akhul", "subject": "Social Studies"}
    ]
    
    # Ensure teachers exist in SQLite teachers table
    for t in teachers_list:
        exists = conn.execute("SELECT 1 FROM teachers WHERE id = ?", (t["id"],)).fetchone()
        if not exists:
            # Try to get user_id for this teacher from users table
            user = conn.execute("SELECT id FROM users WHERE username = ?", (t["name"],)).fetchone()
            user_id = user["id"] if user else None
            conn.execute("INSERT OR IGNORE INTO teachers (id, user_id, name, subject) VALUES (?, ?, ?, ?)", (t["id"], user_id, t["name"], t["subject"]))
    conn.commit()
    
    # Clear old generated remarks to prevent duplicate bloat
    print("Clearing database table for fresh premium presentation generation...")
    conn.execute("DELETE FROM teacher_remarks")
    conn.commit()
    
    # High-fidelity templates for realistic teacher remarks
    templates = {
        "Academic": {
            "Positive": [
                "Excellent classroom participation and consistent performance in assessments.",
                "Shows strong interest in Mathematics and analytical logic.",
                "Actively participates in science quizzes and exhibits top-tier potential.",
                "Exemplary research and study habits observed during assignments.",
                "Shows brilliant comprehension of complex topics and asks highly engaging questions."
            ],
            "Average": [
                "Needs improvement in regular assignment submission times.",
                "Can perform significantly better with regular revision of class topics.",
                "Capable of getting top grades; needs to avoid silly mistakes in exam papers.",
                "Needs to participate more actively in peer group discussions.",
                "Adequate classroom response, but can push for higher academic consistency."
            ],
            "Negative": [
                "Poor test performance in Science. Needs intensive revision.",
                "Struggling to keep up with the advanced syllabus; additional study hours recommended.",
                "Frequently neglects to review test feedback, leading to repeated errors.",
                "Needs to put in substantial effort in core science concepts to avoid failing.",
                "Inconsistent class preparation is leading to slipping grades."
            ]
        },
        "Behaviour": {
            "Positive": [
                "Extremely polite, helpful to peers, and maintains an exemplary attitude.",
                "Displays remarkable leadership qualities during collaborative group tasks.",
                "Very respectful to teachers and supportive of fellow classmates.",
                "Brings highly positive energy to the classroom every single day.",
                "Handles peer disagreements with maturity and emotional intelligence."
            ],
            "Average": [
                "Good overall behavior, but occasionally gets distracted by social circles.",
                "Generally well-behaved, but needs to focus more on active, patient listening.",
                "Polite in nature; needs to develop more confidence when speaking in public.",
                "Requires occasional redirection to remain fully focused on learning activities."
            ],
            "Negative": [
                "Frequently chats during active lectures and disrupts neighboring students.",
                "Needs to demonstrate a more cooperative attitude during team projects.",
                "Exhibits restless behavior and has difficulty maintaining focus during long study blocks.",
                "Needs to follow general classroom etiquette and respect lecture boundaries."
            ]
        },
        "Attendance": {
            "Positive": [
                "Perfect attendance record. Outstanding dedication and daily consistency.",
                "Always punctual and fully prepared for early morning lessons.",
                "Highly reliable attendance; has not missed a single crucial review class.",
                "Maintains immaculate punctuality and values class time.",
                "Attendance has improved significantly this month."
            ],
            "Average": [
                "Attendance should improve. Has missed several crucial revision days.",
                "Occasional late arrivals to early morning classes without valid reasons.",
                "Needs to be more consistent with attendance to avoid learning gaps.",
                "Missed a couple of exam preparation sessions, needs to cover up."
            ],
            "Negative": [
                "Frequently absent from class without prior notification or medical certificates.",
                "Severe attendance shortage is significantly impacting overall marks.",
                "Habitually late to first hour lectures, resulting in missed introductory material.",
                "Extremely poor attendance record; immediate parent conference required."
            ]
        },
        "Discipline": {
            "Positive": [
                "Impeccable discipline, always adheres to dress code and school ethics.",
                "Maintains high integrity and acts as a model student for the entire batch.",
                "Values rules and contributes positively to classroom decorum.",
                "Extremely disciplined approach to both learning and peer relationships."
            ],
            "Average": [
                "Needs to follow school uniform guidelines more strictly.",
                "Occasionally late to morning assembly; punctuality needs correction.",
                "Needs to minimize side conversations during individual study times."
            ],
            "Negative": [
                "Repeatedly late to lectures without valid excuses.",
                "Disciplinary actions may be taken if classroom disruptions continue.",
                "Shows disregard for homework deadlines; needs strict monitoring at home.",
                "Using mobile devices or distractions during active lecture hours."
            ]
        },
        "Performance": {
            "Positive": [
                "Outstanding exam scores and highly motivated toward academic excellence.",
                "Shows consistent, commendable upward trend in all recent internal tests.",
                "Excellent grasp of topics, scores top-tier marks in project reviews.",
                "One of the top achievers in assignments and theoretical exams."
            ],
            "Average": [
                "Performance is stable, but possesses capability for much higher milestones.",
                "Adequate test results; needs more structured self-study patterns.",
                "Performance fluctuates; requires more focus on weak subject chapters.",
                "Shows potential, but lacks dedication to convert it into high scores."
            ],
            "Negative": [
                "Grades are slipping significantly; immediate study revision program is necessary.",
                "Underperformed in recent assessments; needs rigorous remedial support.",
                "Homework and project scores are far below expectations; action required."
            ]
        },
        "Homework": {
            "Positive": [
                "Shows excellent consistency in homework submission.",
                "Homework is always exceptionally detailed, neat, and submitted ahead of time.",
                "Consistently puts in extra effort to present homework flawlessly.",
                "Brilliant accuracy and understanding shown in daily homework assignments."
            ],
            "Average": [
                "Generally completes homework, but sometimes lacks detail and thoroughness.",
                "Needs to submit homework on time to avoid grade penalties.",
                "Homework quality is satisfactory, but can be improved with more effort."
            ],
            "Negative": [
                "Frequently fails to submit homework on time.",
                "Homework is often incomplete or missing key sections.",
                "Severe lack of effort in homework preparation; parents must monitor daily submissions."
            ]
        },
        "Participation": {
            "Positive": [
                "Actively participates in group discussions.",
                "Demonstrates leadership qualities during activities.",
                "Always eager to participate in classroom activities and hands-on experiments.",
                "Energetic contributor to all class events and peer learning projects."
            ],
            "Average": [
                "Participates when called upon, but should volunteer more frequently.",
                "Needs to engage more actively in collaborative group discussions.",
                "Quiet in class; has good ideas but needs encouragement to share them."
            ],
            "Negative": [
                "Rarely participates or contributes to class discussions.",
                "Avoids group work and classroom activities; needs to build collaborative habits.",
                "Prefers to remain disengaged during peer learning sessions."
            ]
        },
        "General Feedback": {
            "Positive": [
                "A true pleasure to have in class. Keep up the brilliant, inspiring work!",
                "An outstanding all-rounder who balances academics and personal growth beautifully.",
                "Possesses a very bright future; shows all-around excellence.",
                "A highly promising talent with exceptional problem-solving abilities."
            ],
            "Average": [
                "A good student with solid potential; needs to maintain consistency.",
                "On the right path, but needs more confidence during final mock presentations.",
                "With slightly more effort, can easily move from Average to Safe performance."
            ],
            "Negative": [
                "Immediate parent-teacher consultation is highly advised to address critical gaps.",
                "Needs to put in substantial after-hours study to secure passing grades.",
                "Lacks motivation in class; parent support is crucial to help them refocus."
            ]
        }
    }
    
    bulk_data = []
    student_ids = list(students_lookup.keys())
    
    print(f"Generating realistic remarks for {len(student_ids)} students...")
    
    # Seed for deterministic yet realistic randomness
    random.seed(1337)
    
    base_date = datetime.datetime.now() - datetime.timedelta(days=120)
    
    for sid in student_ids:
        s_info = students_lookup[sid]
        
        # 1. Assign the "written_by" field based on class exactly:
        # 8th class -> written_by = "nishu" (id 2)
        # 9th class -> written_by = "Akhul" (id 6)
        # 10th class -> written_by = "Roopa" (id 1)
        s_class_str = str(s_info["class"]).lower()
        if "8" in s_class_str:
            t_id = 2
            t_name = "nishu"
        elif "9" in s_class_str:
            t_id = 6
            t_name = "Akhul"
        else:
            t_id = 1
            t_name = "Roopa"
        
        # Between 3 to 10 remarks per student
        num_remarks = random.randint(3, 10)
        
        # Quality distribution based on student's ID so it remains deterministic but varied
        profile_seed = random.random()
        if profile_seed < 0.25:
            quality_pref = "Positive"
        elif profile_seed < 0.85:
            quality_pref = "Average"
        else:
            quality_pref = "Negative"
            
        used_templates = set()
        
        for r_idx in range(num_remarks):
            # Pick a unique category and template to prevent repetitions
            categories_pool = ["Academic", "Behaviour", "Attendance", "Discipline", "Performance", "Homework", "Participation", "General Feedback"]
            category = categories_pool[r_idx % len(categories_pool)]
            
            # Slightly randomize quality around preference
            roll = random.random()
            if roll < 0.70:
                quality = quality_pref
            elif roll < 0.90:
                quality = "Average"
            else:
                quality = "Positive" if quality_pref == "Average" else "Negative" if quality_pref == "Positive" else "Positive"
                
            templates_list = templates[category][quality]
            remark_content = random.choice(templates_list)
            
            # Ensure uniqueness
            attempt = 0
            while remark_content in used_templates and attempt < 10:
                remark_content = random.choice(templates_list)
                attempt += 1
            used_templates.add(remark_content)
            
            # Priority mapping
            if quality == "Negative":
                priority = "High"
            elif quality == "Average":
                priority = "Medium"
            else:
                priority = "Low"
                
            # Rating mapping (1 to 5 stars)
            if quality == "Positive":
                rating = random.randint(4, 5)
            elif quality == "Average":
                rating = random.randint(3, 4)
            else:
                rating = random.randint(1, 2)
                
            # Randomize date over past 120 days
            days_offset = random.randint(0, 120)
            hours_offset = random.randint(8, 16)
            minutes_offset = random.randint(0, 59)
            seconds_offset = random.randint(0, 59)
            
            remark_date = base_date + datetime.timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset, seconds=seconds_offset)
            date_str = remark_date.strftime("%Y-%m-%d %H:%M:%S")
            
            bulk_data.append((
                sid,
                s_info["name"],
                s_info["class"],
                s_info["section"],
                t_id,
                t_name,
                remark_content,
                category,
                priority,
                rating,
                date_str
            ))
            
    # Bulk insert into teacher_remarks
    print(f"Bulk inserting {len(bulk_data)} remarks into SQLite Database...")
    conn.executemany("""
        INSERT INTO teacher_remarks 
        (student_code, student_name, class, section, teacher_id, teacher_name, remark, category, priority, rating, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, bulk_data)
    
    conn.commit()
    conn.close()
    print("Database populate & generation completed successfully!")

if __name__ == "__main__":
    run_migration_and_generation()
