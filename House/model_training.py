# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load data
train_data = pd.read_csv("train.csv")

# Select features (add more if needed)
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Drop missing values
train_data = train_data[features + [target]].dropna()

# Split data
X = train_data[features]
y = train_data[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = GradientBoostingRegressor()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved.")
