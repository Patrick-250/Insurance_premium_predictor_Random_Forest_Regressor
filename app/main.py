from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import pickle
from pydantic import BaseModel
import pandas as pd
from pathlib import Path


app=FastAPI()


with open("./model/random-forest_model.pkl","rb") as model_file:
  model=pickle.load(model_file)

README_PATH=Path(__file__).parent.parent / "README.md"

class DriverProfile(BaseModel):
    age: int
    location_type: str
    driving_points: int
    years_driving_experience: int
    vehicle_type: str
    annual_mileage: float
    credit_score: int
    previous_claims: int
    marital_status: str

@app.post("/predict")
def predict_price(request: DriverProfile):
    try:
        df = pd.DataFrame([{
        "age": request.age,
        "location_type": request.location_type,
        "driving_points": request.driving_points,
        "years_driving_experience": request.years_driving_experience,
        "vehicle_type": request.vehicle_type,
        "annual_mileage": request.annual_mileage,
        "credit_score": request.credit_score,
        "previous_claims": request.previous_claims,
        "marital_status": request.marital_status
    }])

  
        premium_price = model.predict(df)

        return {"premium_amount": float(premium_price[0])}
    except Exception as e:
        print(str(e))
        return "Sorry, there was error while trying to get you the premium.. please call our customer services"
   
@app.get("/", response_class=PlainTextResponse)
def get_info():
    if README_PATH.exists():
        return README_PATH.read_text(encoding="utf-8")
    else:
        return "Insurance Premium Predictor"
