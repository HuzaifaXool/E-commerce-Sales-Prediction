import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt

os.makedirs('processed_data', exist_ok=True)
os.makedirs('pkl_files', exist_ok=True)

oe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()
linear_model = LinearRegression()

def load_data(file_path):
    return pd.read_csv(file_path)

def data_transformation(data):
    encoded_data = oe.fit_transform(data[['Neighborhood']])
    encoded_df = pd.DataFrame(encoded_data, columns=oe.get_feature_names_out(['Neighborhood']))
    data = pd.concat([data, encoded_df], axis=1)
    data.drop(columns=['Neighborhood'], inplace=True)
    return data

def train_split(data):
    x = data.drop(columns=['Price'])
    y = data['Price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

def save_processed_data(x_train, y_train, x_test, y_test):
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)
    train_data.to_csv('processed_data/train.csv', index=False, encoding='utf-8')
    test_data.to_csv('processed_data/test.csv', index=False, encoding='utf-8')

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse_val = mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse_val:.4f}')
    print(f'Mean Absolute Error: {mae_val:.4f}')
    print(f'R2 Score: {r2_val:.4f}')
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs Predicted for {model.__class__.__name__}')
    plt.show()

def save_model(model, file_name):
    joblib.dump(model, file_name)

def predict_price(new_data, model_path, encoder_path, scaler_path):
    oe = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    encoded_data = oe.transform(new_data[['Neighborhood']])
    encoded_df = pd.DataFrame(encoded_data, columns=oe.get_feature_names_out(['Neighborhood']))
    new_data = pd.concat([new_data, encoded_df], axis=1)
    new_data.drop(columns=['Neighborhood'], inplace=True)
    scaled_data = scaler.transform(new_data)
    predictions = model.predict(scaled_data)
    return predictions

if __name__ == "__main__":
    data_file = 'data/housing_price_dataset.csv'
    data = load_data(data_file)
    data = data_transformation(data)
    x_train, y_train, x_test, y_test = train_split(data)
    save_processed_data(x_train, y_train, x_test, y_test)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    train_and_evaluate(linear_model, x_train_scaled, y_train, x_test_scaled, y_test)
    save_model(oe, 'pkl_files/onehot_encoder.pkl')
    save_model(scaler, 'pkl_files/scaler.pkl')
    save_model(linear_model, 'pkl_files/linear_model.pkl')
