# Insurance Premium Predictor

A machine learning model that predicts insurance premiums based on driver demographics, vehicle information, and driving history. This project uses a **Random Forest Regressor** for high-accuracy predictions.

## Features

- Predict insurance premiums based on:
  - Age
  - Location type (Urban/Suburban/Rural)
  - Driving points
  - Years of driving experience
  - Vehicle type (SUV, Sedan, Truck, etc.)
  - Annual mileage
  - Credit score
  - Previous claims
  - Marital status
- Handles both categorical and numerical data
- Scalable for additional features or datasets
- Easy-to-use Python implementation

## Dataset

The dataset includes the following columns:

| Column | Description |
|--------|-------------|
| age | Driver’s age in years |
| location_type | Urban, Suburban, or Rural |
| driving_points | Points on driving record |
| years_driving_experience | Number of years driving |
| vehicle_type | Type of vehicle (SUV, Sedan, Truck) |
| annual_mileage | Annual miles driven |
| credit_score | Credit score of the driver |
| previous_claims | Number of previous insurance claims |
| marital_status | Married or Single |
| policy_premium | Insurance premium in USD |

## Model

- Algorithm: `RandomForestRegressor` from scikit-learn
- Trained to minimize Mean Squared Error (MSE) and maximize prediction accuracy
- Handles feature importance to understand key factors influencing premiums

## requirements
Requirements

Python 3.8+

scikit-learn

pandas

numpy

## Installation

```bash
git clone https://github.com/Patrick-250/Insurance_premium_predictor.git
cd app
pip install -r requirements.txt # can optionaly create virtual environment for dependencies...