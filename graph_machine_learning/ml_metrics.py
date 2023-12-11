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
import multiprocessing
import threading
import queue
import csv
from sklearn.multioutput import MultiOutputRegressor


output_file = 'model_results.csv'

# 1. Read CSV file with data
df = pd.read_csv('aggregate_results.csv')

df.fillna(0, inplace=True)

# Assuming the last columns are the target variables
X = df.iloc[:, :-4]  # All columns except the last four
y = df.iloc[:, -4:]  # Last four columns as targets

print('\n\n\n X \n')
print(X.head(5))
print('\n\n\n Y \n')
print(y.head(5))
print('\n\n')

# 2. Split training and testing data based on 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print('completed splitting data')
# Normalize the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
print('completed normalizing dataset')
# If your target variables also need scaling (depends on your problem)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
print('completed normalizing label dataset')
# 3. Training the models

models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': MultiOutputRegressor(GradientBoostingRegressor()),
    'ElasticNet': MultiOutputRegressor(ElasticNet()),
    # Make sure to set the appropriate objective for XGBoost
    'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
    'LGBMRegressor': LGBMRegressor(),
    # SVR requires special handling for multi-output
    'MultiOutputSVR': MultiOutputRegressor(SVR())
}


def write_result(q: queue.Queue, lock):
    while True:
        data = q.get()
        if data is None:
            break
        write_to_csv(data, lock)

        q.task_done()


def write_to_csv(data, lock):
    with lock:
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(list(data))


def training_model(name, model, lock):
    q = queue.Queue()

    writer = threading.Thread(target=write_result, args=(q, lock))
    writer.start()

    print(f'training {name}')
    try:
        model.fit(X_train_scaled, y_train_scaled)
        print(f'completed fitting {name}')
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        print(f'completed {name} y_train and y_test prediction')
        # Inverse transform the predictions for error calculation
        y_train_pred = scaler_y.inverse_transform(y_train_pred)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        print(f'completed inversing normalization on {name}')
        train_error = mean_squared_error(y_train, y_train_pred)
        print(f'computed train error on {name}')
        test_error = mean_squared_error(y_test, y_test_pred)
        print(f'computed test error on {name}')
        overfitting = "Yes" if train_error < test_error else "No"

        q.put([name, train_error, test_error, overfitting])

    except Exception as e:
        print(f"ERROR WHILE TRAIING {name}")
        print(e)

    q.put(None)
    writer.join()


if __name__ == '__main__':

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Training Error',
                            'Testing Error', 'Overfitting'])

        num_processes = len(models.keys())

        print('about to start training models...')
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(training_model, [
                         (model_name, models[model_name], lock) for model_name in models.keys()])
