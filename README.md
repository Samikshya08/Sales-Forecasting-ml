Walmart Weekly Sales Prediction 📊
📌 Project Overview:

This project uses Machine Learning to predict Walmart's weekly sales using historical sales data and related features such as store information, holidays, temperature, fuel prices, CPI, unemployment, and date.

Two regression algorithms are implemented and compared:
1. Random Forest Regressor
2. XGBoost Regressor

The models are evaluated using MAE, MSE, RMSE, and R² score to understand their prediction performance.

🎯 Project Objectives:
- Analyze Walmart weekly sales data.
- Perform data preprocessing and feature engineering.
- Extract day, month, and year from the date column.
- Train Random Forest and XGBoost regression models.
- Evaluate and compare model performance.
- Predict weekly sales using sample input data.

🛠️ Technologies Used:
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Machine Learning
- Regression Analysis

📂 Dataset Description:
The project uses WalmartDataSet.csv.

The dataset should contain the following columns:
Column	Description:
Store-	Store identification number
Date-	Date of the sales record
Weekly_Sales-	Weekly sales amount (target variable)
Holiday_Flag-	Indicates whether the week contains a holiday
Temperature-	Temperature during the week
Fuel_Price-	Fuel price
CPI	Consumer- Price Index
Unemployment-	Unemployment rate

The Date column is converted into datetime format, and the following features are extracted:
- Day
- Month
- Year
The original Date column is then removed.

⚙️ Project Workflow:
Load the dataset using Pandas.
Display the first few rows of the dataset.
Check for missing values.
Preprocess the date column.
Extract date-related features.
Separate input features and the target variable.
Split the dataset into training and testing sets.
Apply feature scaling for XGBoost.
Train the Random Forest Regressor.
Train the XGBoost Regressor.
Evaluate both models using regression metrics.
Generate sample weekly sales predictions.

🤖 Machine Learning Models:
1. Random Forest Regressor
- Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.
Parameters used:
- Number of estimators: 100
- Random state: 42
- 
2. XGBoost Regressor
- XGBoost is a gradient boosting algorithm that builds decision trees sequentially to improve predictive performance.
Parameters used:
- Number of estimators: 100
- Learning rate: 0.1
- Maximum depth: 5
- Random state: 42
  
📈 Model Evaluation:
The models are evaluated using the following metrics:
- Metric	Purpose
- MAE	Measures the average absolute prediction error
- MSE	Measures the average squared prediction error
- RMSE	Measures prediction error in the target variable's units
- R² Score	Measures how well the model explains variation in sales
