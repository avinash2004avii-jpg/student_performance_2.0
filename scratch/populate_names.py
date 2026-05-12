import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "data", "students_data.csv")

male_first = [
    "Arjun", "Aditya", "Aarav", "Vihaan", "Rahul", "Amit", "Rohan", "Sai", "Karthik", "Akash",
    "Abhinav", "Nikhil", "Pranav", "Yash", "Dev", "Siddharth", "Varun", "Ayush", "Ishan", "Dhruv",
    "Kabir", "Rithvik", "Ansh", "Shaurya", "Tejas", "Harsh", "Vivek", "Sanjay", "Kiran", "Alok",
    "Madhav", "Gautam", "Raghav", "Sameer", "Anand", "Vijay", "Raj", "Sunil", "Sandeep", "Manoj",
    "Abhishek", "Deepak", "Vikram", "Ravi", "Anil", "Suresh", "Ramesh", "Mahesh", "Rajesh", "Dinesh"
]

female_first = [
    "Ananya", "Diya", "Priya", "Sneha", "Neha", "Riya", "Aaradhya", "Ishita", "Kavya", "Shreya",
    "Tanvi", "Pooja", "Shruti", "Meera", "Aditi", "Divya", "Mehak", "Aisha", "Kritika", "Swati",
    "Jyoti", "Deepika", "Priyanka", "Nisha", "Shalini", "Sonia", "Payal", "Preeti", "Rekha", "Kiran",
    "Gita", "Radha", "Seema", "Sunita", "Lakshmi", "Swetha", "Harini", "Keerthana", "Meghana", "Navya",
    "Aishwarya", "Amrutha", "Anusha", "Bhavana", "Chaithra", "Divya", "Kavitha", "Manjula", "Nandini", "Roopa"
]

last_names = [
    "Kumar", "Sharma", "Patel", "Nair", "Singh", "Joshi", "Rao", "Reddy", "Gupta", "Verma",
    "Iyer", "Shastry", "Murthy", "Bhat", "Patil", "Deshmukh", "Gowda", "Hegde", "Shetty", "Pai",
    "Prasad", "Achar", "Acharya", "Choudhury", "Das", "Sen", "Roy", "Banerjee", "Mukherjee", "Chatterjee"
]

def populate():
    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        return
        
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded CSV with {len(df)} rows.")
    
    # Check if Gender exists, if not we assume neutral
    genders = df["Gender"].fillna("Male").tolist() if "Gender" in df.columns else ["Male"] * len(df)
    
    names = []
    for idx, row in df.iterrows():
        # Use existing name if present and not null/NA
        if "name" in df.columns and pd.notna(row["name"]) and str(row["name"]).strip() != "" and str(row["name"]).strip() != "N/A":
            names.append(row["name"])
            continue
            
        gender = str(row.get("Gender", "Male")).strip().capitalize()
        
        # Use Student_ID or index to deterministically assign names so they don't change randomly
        sid = int(row.get("Student_ID", idx))
        
        if gender == "Female":
            first = female_first[sid % len(female_first)]
        else:
            first = male_first[sid % len(male_first)]
            
        last = last_names[(sid * 7) % len(last_names)]
        names.append(f"{first} {last}")
        
    df["name"] = names
    
    # Reorder columns to put name after Student_ID
    cols = list(df.columns)
    if "name" in cols:
        cols.remove("name")
    insert_idx = cols.index("Student_ID") + 1 if "Student_ID" in cols else 0
    cols.insert(insert_idx, "name")
    df = df[cols]
    
    df.to_csv(DATA_FILE, index=False)
    print("CSV successfully populated with student names!")

if __name__ == "__main__":
    populate()
