import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import traceback
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def fetch_and_preprocess_data():
    print("Fetching and preprocessing data...")
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    data = pd.read_csv(url)
    data['Date_reported'] = pd.to_datetime(data['Date_reported'])
    global_data = data.groupby('Date_reported').agg({'New_cases': 'sum'}).reset_index()
    global_data['New_cases'] = np.log1p(global_data['New_cases'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    global_data['New_cases'] = scaler.fit_transform(global_data['New_cases'].values.reshape(-1, 1))
    return global_data, scaler

def create_features(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    X, y = np.array(X), np.array(y)
    X_flat = X.reshape(X.shape[0], -1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, X_flat, y

def create_lstm_gru_model(sequence_length):
    model = Sequential([
        Input(shape=(sequence_length, 1)),
        GRU(40, return_sequences=True),
        Dropout(0.1),
        LSTM(20, return_sequences=True),
        BatchNormalization(),
        GRU(90),
        BatchNormalization(),
        Dropout(0.4),
        Dense(40, activation='relu'),
        Dense(100, activation='relu'),
        Dense(70, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007919746988842461),
                  loss='mean_absolute_percentage_error')
    return model

def train_and_predict_lstm_gru(X_train, y_train, X_test):
    model = create_lstm_gru_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=60, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=2)
    predictions = model.predict(X_test)
    return predictions.flatten()

def train_and_predict_arima(train, test_length):
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=test_length)
    return predictions

def train_and_predict_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def train_and_predict_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def main():
    try:
        print("Starting main function...")
        global_data, scaler = fetch_and_preprocess_data()
        sequence_length = 90
        prediction_length = 7 * 7  # 7 weeks

        X, X_flat, y = create_features(global_data['New_cases'].values, sequence_length)
        
        # Use the last 7 weeks for testing
        X_train, X_test = X[:-prediction_length], X[-prediction_length:]
        X_flat_train, X_flat_test = X_flat[:-prediction_length], X_flat[-prediction_length:]
        y_train, y_test = y[:-prediction_length], y[-prediction_length:]
        
        # Train and predict using LSTM/GRU model
        lstm_gru_predictions = train_and_predict_lstm_gru(X_train, y_train, X_test)
        
        # Train and predict using ARIMA model
        arima_predictions = train_and_predict_arima(global_data['New_cases'].values[:-prediction_length], prediction_length)
        
        # Train and predict using Random Forest
        rf_predictions = train_and_predict_random_forest(X_flat_train, y_train, X_flat_test)
        
        # Train and predict using XGBoost
        xgb_predictions = train_and_predict_xgboost(X_flat_train, y_train, X_flat_test)
        
        # Inverse transform predictions
        lstm_gru_predictions = np.expm1(scaler.inverse_transform(lstm_gru_predictions.reshape(-1, 1)).flatten())
        arima_predictions = np.expm1(scaler.inverse_transform(arima_predictions.reshape(-1, 1)).flatten())
        rf_predictions = np.expm1(scaler.inverse_transform(rf_predictions.reshape(-1, 1)).flatten())
        xgb_predictions = np.expm1(scaler.inverse_transform(xgb_predictions.reshape(-1, 1)).flatten())
        actual = np.expm1(scaler.inverse_transform(y_test.reshape(-1, 1)).flatten())
        
        # Calculate MAPE for each model
        lstm_gru_mape = calculate_mape(actual, lstm_gru_predictions)
        arima_mape = calculate_mape(actual, arima_predictions)
        rf_mape = calculate_mape(actual, rf_predictions)
        xgb_mape = calculate_mape(actual, xgb_predictions)
        
        print(f"LSTM/GRU MAPE: {lstm_gru_mape}")
        print(f"ARIMA MAPE: {arima_mape}")
        print(f"Random Forest MAPE: {rf_mape}")
        print(f"XGBoost MAPE: {xgb_mape}")
        
        # Prepare data for JSON output
        last_date = global_data['Date_reported'].iloc[-1]
        prediction_dates = [last_date + timedelta(days=7*i) for i in range(7)]
        
        data = {
            'dates': [d.strftime('%Y-%m-%d') for d in prediction_dates],
            'actual': actual.tolist(),
            'lstm_gru_predicted': lstm_gru_predictions.tolist(),
            'arima_predicted': arima_predictions.tolist(),
            'rf_predicted': rf_predictions.tolist(),
            'xgb_predicted': xgb_predictions.tolist(),
            'lstm_gru_mape': float(lstm_gru_mape),
            'arima_mape': float(arima_mape),
            'rf_mape': float(rf_mape),
            'xgb_mape': float(xgb_mape),
            'last_updated': datetime.now().isoformat()
        }
        
        print("Saving to JSON...")
        json_path = os.path.join(os.getcwd(), 'covid_predictions.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        print(f"JSON file saved successfully at: {json_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()