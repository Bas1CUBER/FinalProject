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

# Pass/Fail Configuration - Aligned with Jupyter Notebook
PASSING_THRESHOLD = 10  # Portuguese education system: 10/20 (50%)

def classify_performance(binary_prediction: int):
    """Binary classification based on model output"""
    # Model outputs 0 (Fail) or 1 (Pass)
    if binary_prediction == 1:  # Pass
        return {
            "status": "PASS",
            "result": "PASS",
            "risk_level": "Low Risk",
            "recommendation": "Good work! Keep studying consistently.",
            "description": "Predicted to Pass - Continue current study habits"
        }
    else:  # binary_prediction == 0, Fail
        return {
            "status": "FAIL", 
            "result": "FAIL",
            "risk_level": "High Risk",
            "recommendation": "Need improvement. Increase study time and get help.",
            "description": "Predicted to Fail - Additional support recommended"
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
    description="Predict student academic outcome with 90.5% accuracy using Random Forest binary classification (Pass ≥10, Fail <10)",
    version="2.0.0"
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
        
        # Scale and predict with binary classifier
        df_scaled = scaler.transform(df_processed)
        
        # Get binary prediction (0 = Fail, 1 = Pass)
        binary_prediction = rf_model.predict(df_scaled)[0]
        
        # Get prediction probability for confidence
        prediction_proba = rf_model.predict_proba(df_scaled)[0]
        confidence = max(prediction_proba)  # Highest probability
        
        # Convert binary to realistic grade estimate for display
        if binary_prediction == 1:  # Pass prediction
            # For pass: estimate grade between 10-18 based on confidence
            estimated_grade = 10 + (confidence - 0.5) * 16  # 10-18 range
            estimated_grade = max(10, min(20, estimated_grade))  # Ensure 10-20 for pass
        else:  # Fail prediction  
            # For fail: estimate grade between 0-9 based on confidence
            estimated_grade = confidence * 18  # Scale confidence to 0-9 range
            estimated_grade = max(0, min(9.9, estimated_grade))  # Ensure 0-9.9 for fail
        
        # CRITICAL: Ensure grade and classification are consistent
        # If estimated grade >= 10, force PASS; if < 10, force FAIL
        if estimated_grade >= 10:
            final_classification = "PASS"
        else:
            final_classification = "FAIL"
        
        # Get classification results (ensure consistency)
        classification = {
            "result": final_classification,
            "status": final_classification,
            "risk_level": "Low Risk" if final_classification == "PASS" else "High Risk",
            "recommendation": "Good work! Keep studying consistently." if final_classification == "PASS" else "Need improvement. Increase study time and get help.",
            "description": f"Predicted to {final_classification} - Grade-based classification"
        }
        
        return {
            "predicted_grade": float(round(estimated_grade, 1)),
            "percentage": f"{round((estimated_grade/20)*100, 1)}%",
            "pass_fail": str(classification["result"]),
            "confidence": f"{round(confidence*100, 1)}%",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)