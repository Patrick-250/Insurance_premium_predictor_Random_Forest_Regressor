import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("./policy_premium_dataset.csv")

# Features & target
X = data.drop("policy_premium", axis=1)
y = data["policy_premium"]
print(y.mean())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify categorical columns to encode them
categorical_columns = ["location_type", "marital_status","vehicle_type"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# Create pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestRegressor(n_estimators=20))
])

# Train
model.fit(X_train, y_train)

# Predict
predicted = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predicted)
rmse=np.sqrt(mse)
print("rmse:", rmse)

# try:
#   with open("random-forest_model.pkl","wb") as model_pkl:
#     pickle.dump(model,model_pkl)
#     print("random forest model sucessfully saved..")
# except Exception as e:
#   print("failed to pickle model")

