from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load
from typing import Dict, Iterator

# Load trained pipeline (preprocessor + logistic regression)
model = load("artifacts/model.joblib")

# Define FastAPI app
app = FastAPI(title="Thyroid Cancer Recurrence Predictor")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def get_root():
    return FileResponse("static/index.html")

# Define input schema using Pydantic
class PatientData(BaseModel):
    Age: int
    Gender: str
    Smoking: str
    HxSmoking: str = Field(..., alias="Hx Smoking")
    HxRadiotherapy: str = Field(..., alias="Hx Radiotherapy")
    Thyroid_Function: str = Field(..., alias="Thyroid Function")
    Physical_Examination: str = Field(..., alias="Physical Examination")
    Adenopathy: str
    Pathology: str
    Focality: str
    Risk: str
    T: str
    N: str
    M: str
    Stage: str
    Response: str

# Base class with generator
class BasePredictor:
    def data_generator(self, row: Dict) -> Iterator[Dict]:
        # Explicit column mapping to match training pipeline
        yield {
            "Age": row["Age"],
            "Gender": row["Gender"],
            "Smoking": row["Smoking"],
            "Hx Smoking": row["HxSmoking"],
            "Hx Radiothreapy": row["HxRadiotherapy"],  # typo preserved
            "Thyroid Function": row["Thyroid_Function"],
            "Physical Examination": row["Physical_Examination"],
            "Adenopathy": row["Adenopathy"],
            "Pathology": row["Pathology"],
            "Focality": row["Focality"],
            "Risk": row["Risk"],
            "T": row["T"],
            "N": row["N"],
            "M": row["M"],
            "Stage": row["Stage"],
            "Response": row["Response"]
        }

# Child class: predictor for thyroid cancer recurrence
class ThyroidPredictor(BasePredictor):
    def predict(self, data: PatientData):
        df = pd.DataFrame(list(self.data_generator(data.dict())))
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return {
            "prediction": {0: "No", 1: "Yes"}[prediction],
            "probability": f"{probability*100:.2f}%"
        }

# Instantiate predictor
predictor = ThyroidPredictor()

# FastAPI endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        return predictor.predict(data)
    except Exception as e:
        return {"error": str(e)}

# For local testing
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
