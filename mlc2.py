import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ============================================================
# LOAD DATASET
# ============================================================

# Read CSV File
df = pd.read_csv("WalmartDataSet.csv")

# Display Dataset
print("DATASET SAMPLE\n")

print(df.head())

# ============================================================
# DATA PREPROCESSING
# ============================================================

# Convert Date Column
# Dataset format = DD-MM-YYYY

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y"
)

# Extract Date Features

df["Day"] = df["Date"].dt.day

df["Month"] = df["Date"].dt.month

df["Year"] = df["Date"].dt.year

# Drop Original Date Column

df.drop("Date", axis=1, inplace=True)

# Check Missing Values

print("\nMISSING VALUES\n")

print(df.isnull().sum())

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop("Weekly_Sales", axis=1)

y = df["Weekly_Sales"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ============================================================
# XGBOOST REGRESSION
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

xgb_predictions = xgb_model.predict(X_test_scaled)

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n===================================")
print("RANDOM FOREST REGRESSION RESULTS")
print("===================================")

print("MAE:",
      mean_absolute_error(y_test, rf_predictions))

print("MSE:",
      mean_squared_error(y_test, rf_predictions))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, rf_predictions)))

print("R2 Score:",
      r2_score(y_test, rf_predictions))

# ============================================================

print("\n===================================")
print("XGBOOST REGRESSION RESULTS")
print("===================================")

print("MAE:",
      mean_absolute_error(y_test, xgb_predictions))

print("MSE:",
      mean_squared_error(y_test, xgb_predictions))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, xgb_predictions)))

print("R2 Score:",
      r2_score(y_test, xgb_predictions))

# ============================================================
# SAMPLE PREDICTION
# ============================================================

sample_data = pd.DataFrame([{
    "Store": 1,
    "Holiday_Flag": 0,
    "Temperature": 42.31,
    "Fuel_Price": 2.57,
    "CPI": 211.10,
    "Unemployment": 8.10,
    "Day": 5,
    "Month": 2,
    "Year": 2010
}])

# Random Forest Prediction

rf_sales = rf_model.predict(sample_data)

# XGBoost Prediction

sample_scaled = scaler.transform(sample_data)

xgb_sales = xgb_model.predict(sample_scaled)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n===================================")
print("FUTURE SALES PREDICTION")
print("===================================")

print("Random Forest Prediction:",
      round(rf_sales[0], 2))

print("XGBoost Prediction:",
      round(xgb_sales[0], 2))
