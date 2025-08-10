from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load

# Load the model (pipeline: preprocessor + logistic regression)
model = load("artifacts/model.joblib")

# Define the FastAPI app
app = FastAPI(title="Thyroid Cancer Recurrence Predictor")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# @app.get("/")
# def get_root():
#     return FileResponse("static/index.html")


# Define the input schema using Pydantic
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

@app.post("/predict")
def predict(data: PatientData):
    input_dict = data.dict()

    # Rename keys to match model input
    input_renamed = {
        'Age': input_dict['Age'],
        'Gender': input_dict['Gender'],
        'Smoking': input_dict['Smoking'],
        'Hx Smoking': input_dict['HxSmoking'],
        'Hx Radiothreapy': input_dict['HxRadiotherapy'],  # fix spelling if needed
        'Thyroid Function': input_dict['Thyroid_Function'],
        'Physical Examination': input_dict['Physical_Examination'],
        'Adenopathy': input_dict['Adenopathy'],
        'Pathology': input_dict['Pathology'],
        'Focality': input_dict['Focality'],
        'Risk': input_dict['Risk'],
        'T': input_dict['T'],
        'N': input_dict['N'],
        'M': input_dict['M'],
        'Stage': input_dict['Stage'],
        'Response': input_dict['Response']
    }

    input_df = pd.DataFrame([input_renamed])

    # Predict
    try:
        prediction = model.predict(input_df)[0]
        decision = {0: "No", 1: "Yes"}
        probability = model.predict_proba(input_df)[0]
        return {"prediction": decision[prediction], "probability": f"{probability[1]*100:.2f}%"}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
