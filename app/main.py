from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

app = FastAPI()

# Load your trained model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the input data model
class StudentData(BaseModel):
    exam_scores_1: float
    exam_scores_2: float
    exam_scores_3: float
    exam_scores_4: float
    exam_scores_5: float
    exam_scores_6: float
    attendance: float
    extracurricular: int
    study_hours: float
    family_income: float
    school_type: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Prediction API!"}

@app.post("/predict")
def predict(student: StudentData):
    # Ensure the input data is in a DataFrame or match the fitted scaler's expected structure
    input_data_df = pd.DataFrame({
        'exam_scores_1': [student.exam_scores_1],
        'exam_scores_2': [student.exam_scores_2],
        'exam_scores_3': [student.exam_scores_3],
        'exam_scores_4': [student.exam_scores_4],
        'exam_scores_5': [student.exam_scores_5],
        'exam_scores_6': [student.exam_scores_6],
        'attendance': [student.attendance],
        'extracurricular': [student.extracurricular],
        'study_hours': [student.study_hours],
        'family_income': [student.family_income],
        'school_type': [student.school_type]
    })

    # Apply scaling to the DataFrame
    input_data_scaled = scaler.transform(input_data_df)

    # Get the prediction from the trained model
    prediction = model.predict(input_data_scaled)

    # Return the prediction as a JSON response
    return {"prediction": int(prediction[0])}
