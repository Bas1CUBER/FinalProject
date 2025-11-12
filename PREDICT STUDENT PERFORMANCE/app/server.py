from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
# Minimal logging setup

# These are the original 32 features (excluding G3 target)
ORIGINAL_FEATURES = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
    'absences', 'G1', 'G2'
]

# Categorical columns that need LabelEncoder
CATEGORICAL_COLS = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 
    'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
    'higher', 'internet', 'romantic'
]

# Pass/Fail Configuration
PASSING_THRESHOLD = 12  # International standard: 12/20 (60%)

def classify_performance(predicted_grade: float):
    """Simple Pass/Fail classification"""
    grade = round(predicted_grade, 1)
    
    if grade >= PASSING_THRESHOLD:
        return {
            "status": "PASS",
            "result": "PASS",
            "risk_level": "Low Risk" if grade >= 16 else "Moderate Risk",
            "recommendation": "Good work! Keep studying consistently.",
            "description": f"Grade {grade}/20 - Pass"
        }
    else:
        return {
            "status": "FAIL", 
            "result": "FAIL",
            "risk_level": "High Risk",
            "recommendation": "Need improvement. Increase study time and get help.",
            "description": f"Grade {grade}/20 - Fail"
        }

# Load only the 3 essential files: rf_model, scaler, and label_encoder
try:
    # Load Random Forest model (primary model)
    rf_model = joblib.load("models/rf_model.pkl")
    
    # Load scaler 
    scaler = joblib.load("models/scaler.pkl")
    
    # Load label encoder
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    rf_model = None
    scaler = None
    label_encoder = None

class StudentData(BaseModel):
    # Essential inputs
    age: int = 17
    studytime: int = 2 
    failures: int = 0
    absences: int = 6
    G1: int = 5  
    G2: int = 6
    
    # Optional with defaults
    school: str = "GP"
    sex: str = "F" 
    address: str = "U"
    famsize: str = "GT3"
    Pstatus: str = "T"
    Medu: int = 4
    Fedu: int = 4
    Mjob: str = "at_home"
    Fjob: str = "teacher"
    reason: str = "course"
    guardian: str = "mother"
    traveltime: int = 2
    schoolsup: str = "yes"
    famsup: str = "no"
    paid: str = "no"
    activities: str = "no"
    nursery: str = "yes"
    higher: str = "yes"
    internet: str = "no"
    romantic: str = "no"
    famrel: int = 4
    freetime: int = 3
    goout: int = 4
    Dalc: int = 1
    Walc: int = 1
    health: int = 3

# Initialize FastAPI app
app = FastAPI(
    title="GradePilot",
    description="Predict final academic outcome with Pass/Fail analysis using Random Forest model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for CSS and JS)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML interface"""
    with open("assets/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/predict")
async def get_input_schema():
    """Get default student data schema for input"""
    return {
        "age": 17,
        "studytime": 2,
        "failures": 0,
        "absences": 6,
        "G1": 5,
        "G2": 6,
        "school": "GP",
        "sex": "F", 
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "at_home",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "no",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 4,
        "Dalc": 1,
        "Walc": 1,
        "health": 3
    }



@app.post("/predict")
def predict_student(data: StudentData):
    """
    Predict student's final grade and classify as pass/fail
    """
    try:
        # Check if model files are loaded
        if rf_model is None or scaler is None or label_encoder is None:
            raise HTTPException(status_code=500, detail="Model files not loaded properly")
        
        # Process input data
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])[ORIGINAL_FEATURES]
        
        # Apply label encoders
        df_processed = df.copy()
        for col in CATEGORICAL_COLS:
            if col in df_processed.columns:
                try:
                    if isinstance(label_encoder, dict) and col in label_encoder:
                        df_processed[col] = label_encoder[col].transform(df_processed[col].astype(str))
                    else:
                        # Handle categorical encoding manually with simple mapping
                        if col == 'school':
                            df_processed[col] = 1 if df_processed[col].iloc[0] == 'GP' else 0
                        elif col == 'sex':
                            df_processed[col] = 1 if df_processed[col].iloc[0] == 'M' else 0
                        elif col == 'address':
                            df_processed[col] = 1 if df_processed[col].iloc[0] == 'U' else 0
                        elif col in ['famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']:
                            # For other categorical columns, use simple hash-based encoding
                            df_processed[col] = abs(hash(str(df_processed[col].iloc[0]))) % 10
                        elif col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
                            df_processed[col] = 1 if df_processed[col].iloc[0] == 'yes' else 0
                except (ValueError, KeyError):
                    df_processed[col] = 0
        
        # Scale and predict
        df_scaled = scaler.transform(df_processed)
        prediction = float(rf_model.predict(df_scaled)[0])  # Convert to Python float
        prediction = max(0, min(20, prediction))  # Ensure 0-20 range
        
        # Get classification results
        classification = classify_performance(prediction)
        
        return {
            "predicted_grade": float(round(prediction, 1)),
            "percentage": f"{round((prediction/20)*100, 1)}%",
            "pass_fail": str(classification["result"]),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)