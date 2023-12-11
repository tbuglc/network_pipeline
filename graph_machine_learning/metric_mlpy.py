import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

# 1. Read CSV file with data
df = pd.read_csv('aggregate_results.csv ')

# Assuming the last columns are the target variables
X = df.iloc[:, :-4]  # All columns except the last four
y = df.iloc[:, -4:]  # Last four columns as targets

# 2. Split training and testing data based on 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# If your target variables also need scaling (depends on your problem)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 3. Training the models
models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'ElasticNet': ElasticNet(),
    'XGBRegressor': XGBRegressor(),
    'LGBMRegressor': LGBMRegressor(),
    # 'MultiOutputSVR': MultiOutputRegressor(SVR())  # SVR requires special handling for multi-output
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Inverse transform the predictions for error calculation
    y_train_pred = scaler_y.inverse_transform(y_train_pred)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    overfitting = "Yes" if train_error < test_error else "No"

    results.append([name, train_error, test_error, overfitting])

# 4. Save results to CSV
results_df = pd.DataFrame(
    results, columns=['Model', 'Training Error', 'Testing Error', 'Overfitting'])
results_df.to_csv('model_results.csv', index=False)

print("Results saved to model_results.csv")
