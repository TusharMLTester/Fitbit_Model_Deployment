from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Fitbit Machine State Model API", version="1.0")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "sensors_model.pkl")
model = joblib.load(model_path)

class InputData(BaseModel):
    RMS: float
    volt: float
    speed: float

# --------------------------------------------
# Health check route
# --------------------------------------------
@app.get("/")
def home():
    return {"status": "ok", "message": "Fitbit Model API is running"}

#--------------------------------------------
# Prediction one at time
#--------------------------------------------

@app.post("/predict")
def predict_one(data: InputData):
    df = pd.DataFrame([data.model_dump()])
    pred = model.predict(df)[0]
    mapping = {0: "OFF", 1: "IDLE", 2: "ACTIVE"}
    return {"predicted_state": mapping[pred]}
#--------------------------------------------
# Prediction for batch data
#--------------------------------------------

@app.post("/predict_batch")
def predict_batch(data: list[InputData]):
    df = pd.DataFrame([d.model_dump() for d in data])
    preds = model.predict(df).tolist()
    mapping = {0: "OFF", 1: "IDLE", 2: "ACTIVE"}
    return {"predicted_states": [mapping[p] for p in preds]}