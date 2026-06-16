import re
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Percept Infosystem Consultancy- Student Enrollement Prediction API",
    description="""
## 🎓 Percept Infosystem Consultancy
![Logo](percept logo.png)

Predicts whether a student will register based on their course interest.
### Endpoints:
- **POST /predict**-> Single student prediction
- **POST/predict/batch**-> Multiple students prediction
- **GET /purposes** -> List all known courses
""",
    version="1.0.0"
)

# Clean Purpose Function (SAME as notebook and save_artifacts.py)

def clean_purpose(text):

    # Convert to string (handle nan)
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace s/w or s\w with software
    text = re.sub(r's[/\\]w', 'software', text)

    # Replace slash and backslash with ' and '
    text = re.sub(r'[/\\]', ' and ', text)

    # Replace + and & with ' and '
    text = re.sub(r'[+&]', ' and ', text)

    # Remove brackets and special characters except letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Fix common spelling mistakes
    corrections = {
        'intership': 'internship',
        'internnship': 'internship',
        'internishp': 'internship',
        'intenship': 'internship',
        'placment': 'placement',
        'regrding': 'regarding',
        'enqiry': 'enquiry',
        'enquri': 'enquiry',
        'enqurie': 'enquiry',
        'coustomer': 'customer',
        'salse': 'sales',
        'telesalse': 'telesales',
        'markrting': 'marketing',
        'testeing': 'testing',
        'cyberv': 'cyber',
        'concelling': 'counselling'
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load artifacts at startup

pipeline = None
encoder = None
le = None

@app.on_event("startup")
def load_model():
    global pipeline, encoder, le

    if not os.path.exists("pipeline.pkl"):
        raise RuntimeError("pipeline.pkl not found.")
    pipeline = joblib.load("pipeline.pkl")

    if not os.path.exists("encoder.pkl"):
        raise RuntimeError("encoder.pkl not found. Run save_artifacts.py first.")
    encoder = joblib.load("encoder.pkl")

    if not os.path.exists("label_encoder.pkl"):
        raise RuntimeError("label_encoder.pkl not found. Run save_artifacts.py first.")
    le = joblib.load("label_encoder.pkl")

    print(" All models loaded successfully!")


# Schemas

class StudentInput(BaseModel):
    purpose: str

    class Config:
        json_schema_extra = {
            "example": {"purpose": "data science"}
        }

class PredictionResponse(BaseModel):
    purpose: str
    cleaned_purpose: str
    prediction: str
    prediction_label: int
    probability_registered: float
    probability_not_registered: float

class BatchInput(BaseModel):
    students: list[StudentInput]

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]



# Helper

def preprocess_and_predict(purpose: str) -> dict:

    # Step 1: Clean input using same function as notebook
    purpose_clean = clean_purpose(purpose)

    try:
        # Step 2: OneHotEncode the cleaned Purpose
        X_encoded = encoder.transform([[purpose_clean]])

        # Step 3: Pass to pipeline (scaler + model)
        proba = pipeline.predict_proba(X_encoded)[0]
        pred_label = int(pipeline.predict(X_encoded)[0])

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")

    # Step 4: Decode label back to text
    pred_str = le.inverse_transform([pred_label])[0]

    prob_not_reg = round(float(proba[0]), 4)
    prob_reg = round(float(proba[1]), 4) if len(proba) > 1 else round(1 - prob_not_reg, 4)

    return {
        "purpose": purpose,                  # original input
        "cleaned_purpose": purpose_clean,    # after cleaning
        "prediction": pred_str,
        "prediction_label": pred_label,
        "probability_registered": prob_reg,
        "probability_not_registered": prob_not_reg,
    }


# Routes

@app.get("/", tags=["Health"])
def root():
    return {"message": "Student Enrollment Prediction API is running"}

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(student: StudentInput):
    return preprocess_and_predict(student.purpose)

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    if not batch.students:
        raise HTTPException(status_code=400, detail="No students provided.")
    results = [preprocess_and_predict(s.purpose) for s in batch.students]
    return {"results": results}

@app.get("/purposes", tags=["Info"])
def list_known_purposes():
    if encoder:
        return {"known_purposes": list(encoder.categories_[0])}
    return {"known_purposes": []}