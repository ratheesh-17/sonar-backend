from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

from fastapi.middleware.cors import CORSMiddleware


# Load the model
model = joblib.load("logistic_model.pkl")

# Define input schema using Pydantic
class SonarInput(BaseModel):
    values: list[float]  # 60 values in a list

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Sonar ML Model API is running!"}

@app.post("/predict")
def predict(input_data: SonarInput):
    if len(input_data.values) != 60:
        raise HTTPException(status_code=400, detail="Exactly 60 values are required.")

    input_array = np.array(input_data.values).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    return {"prediction": prediction, "message": "The object is Rock." if prediction == 'R' else "The object is Mine."}


