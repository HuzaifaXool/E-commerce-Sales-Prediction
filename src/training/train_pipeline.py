import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import joblib

# Ensure the processed_data directory exists
os.makedirs('processed_data', exist_ok=True)

oe = OneHotEncoder(sparse_output=False)
linear_model = LinearRegression()

data = pd.read_csv('data/housing_price_dataset.csv')

def data_transformation(data):
    encoded_data = oe.fit_transform(data[['Neighborhood']]).astype(int)
    encoded_df = pd.DataFrame(encoded_data, columns=oe.get_feature_names_out(['Neighborhood']))
    data = pd.concat([data, encoded_df], axis=1)
    data.drop(columns=['Neighborhood'], inplace=True)
    return data

def train_split(data):
    x = data.drop(columns=['Price'])
    y = data['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)
    train_data.to_csv('processed_data/train.csv', index=False, encoding='utf-8')
    test_data.to_csv('processed_data/test.csv', index=False, encoding='utf-8')

    return x_train, y_train, x_test, y_test

data = data_transformation(data)
x_train, y_train, x_test, y_test = train_split(data)

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse_val = mse(y_test, y_pred)
    mae_val = mae(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    print(f'Model: {model.__class__.__name__}')
    print(f'Mean Squared Error: {mse_val}')
    print(f'Mean Absolute Error: {mae_val}')
    print(f'R2 Score: {r2_val}')
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted for {model.__class__.__name__}')
    plt.show()

train_and_evaluate(linear_model, x_train, y_train, x_test, y_test)