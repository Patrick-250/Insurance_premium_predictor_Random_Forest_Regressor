from fastapi import FastAPI

app=FastAPI()

@app.get('/')
def home():
  return "predict the insurance premium price... "



@app.post('/predict')
def predict_price():
 
  return " ....calculating..."
  