import pandas as pd
import numpy as np

# Number of students
num_students = 500

# Generate exam scores (6 exams)
exam_scores = np.random.randint(60, 101, size=(num_students, 6))

# Attendance (randomized between 0.7 to 1.0)
attendance = np.random.uniform(0.7, 1.0, num_students)

# Extracurricular participation (random: 0 = No, 1 = Yes)
extracurricular = np.random.choice([0, 1], size=num_students)

# Study hours per week (0-20 hours)
study_hours = np.random.uniform(0, 20, num_students)

# Family income (randomized between 20k to 80k USD)
family_income = np.random.uniform(20000, 80000, num_students)

# School type (0 = private, 1 = public)
school_type = np.random.choice([0, 1], size=num_students)

# Create a DataFrame
data = {
    "exam_scores_1": exam_scores[:, 0],
    "exam_scores_2": exam_scores[:, 1],
    "exam_scores_3": exam_scores[:, 2],
    "exam_scores_4": exam_scores[:, 3],
    "exam_scores_5": exam_scores[:, 4],
    "exam_scores_6": exam_scores[:, 5],
    "attendance": attendance,
    "extracurricular": extracurricular,
    "study_hours": study_hours,
    "family_income": family_income,
    "school_type": school_type
}

df = pd.DataFrame(data)

# Generate labels: (1 = good performance, 0 = poor performance)
# Based on weighted sum of the features to predict performance
df["performance"] = (0.4 * np.mean(exam_scores, axis=1) + 
                     0.2 * attendance * 100 + 
                     0.2 * study_hours + 
                     0.1 * family_income + 
                     0.1 * extracurricular * 10)

# Label students as 'good performance' (1) or 'poor performance' (0)
df["performance"] = np.where(df["performance"] >= 70, 1, 0)

# Save to CSV
df.to_csv('student_performance_data.csv', index=False)
print("Dataset generated and saved to 'student_performance_data.csv'.")
