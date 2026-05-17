from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the trained model and feature columns
model = joblib.load("diabetes_model.pkl")
training_columns = joblib.load("training_columns.pkl")

class PatientData(BaseModel):
    age: float
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: str

@app.get("/")
def home():
    return {"status": "API is running"}
@app.post("/predict")
def predict(data: PatientData):

    # Input dictionary
    input_data = {
        "AGE": data.age,
        "Urea": data.urea,
        "Cr": data.cr,
        "HbA1c": data.hba1c,
        "Chol": data.chol,
        "TG": data.tg,
        "HDL": data.hdl,
        "LDL": data.ldl,
        "VLDL": data.vldl,
        "BMI": data.bmi
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Gender encoding
    if data.gender.upper() == "M":
        df["Gender_M"] = 1
    else:
        df["Gender_M"] = 0

    # Missing columns handling
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Column order match
    df = df[training_columns]

    # Prediction
    prediction = model.predict(df)

    # Result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

    return {
        "prediction": result
    }