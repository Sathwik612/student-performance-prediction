# Student Performance Prediction API

This is an API that predicts a student's future performance based on their past data using a machine learning model. It uses FastAPI to serve the model and allows predictions based on a student's data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Sathwik612/student-performance-prediction.git
   cd student-performance-prediction
2. Install dependencies
   ```bash
   pip install -r app/requirements.txt

   
3. Run the FastAPI server
   ```bash
   uvicorn app.main:app --reload
    
4. You can test the API with Postman or through curl. The input JSON should look like this:
   ```bash
   {
  "exam_scores_1": 85,
  "exam_scores_2": 88,
  "exam_scores_3": 90,
  "exam_scores_4": 92,
  "exam_scores_5": 87,
  "exam_scores_6": 89,
  "attendance": 0.95,
  "extracurricular": 1,
  "study_hours": 15,
  "family_income": 50000,
  "school_type": 1
}




