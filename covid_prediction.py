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

def preprocess_and_prepare_dataset():
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

def create_model(sequence_length):
    model = Sequential()
    model.add(Input(shape=(sequence_length, 1)))
    
    # Layer 0: GRU
    model.add(GRU(40, return_sequences=True))
    model.add(Dropout(0.1))
    
    # Layer 1: LSTM
    model.add(LSTM(20, return_sequences=True))
    model.add(BatchNormalization())
    
    # Layer 2: GRU
    model.add(GRU(90))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Dense layers
    model.add(Dense(40, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(70, activation='relu'))
    
    # Output layer
    model.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007919746988842461)
    model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer)
    
    return model

def train_and_predict_lstm_gru(X, y, X_test):
    model = create_model(X.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X, y, epochs=150, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=2)
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
        global_data, scaler = preprocess_and_prepare_dataset()
        sequence_length = 90  # As per the provided hyperparameters
        
        X, X_flat, y = create_features(global_data['New_cases'].values, sequence_length)
        
        # Use the last 14 days for testing
        test_size = 14
        X_train, X_test = X[:-test_size], X[-test_size:]
        X_flat_train, X_flat_test = X_flat[:-test_size], X_flat[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Train and predict using LSTM/GRU model
        lstm_gru_predictions = train_and_predict_lstm_gru(X_train, y_train, X_test)
        
        # Train and predict using ARIMA model
        arima_predictions = train_and_predict_arima(global_data['New_cases'].values[:-test_size], test_size)
        
        # Train and predict using Random Forest
        rf_predictions = train_and_predict_random_forest(X_flat_train, y_train, X_flat_test)
        
        # Train and predict using XGBoost
        xgb_predictions = train_and_predict_xgboost(X_flat_train, y_train, X_flat_test)
        
        # Calculate MAPE for each model
        lstm_gru_mape = calculate_mape(y_test, lstm_gru_predictions)
        arima_mape = calculate_mape(y_test, arima_predictions)
        rf_mape = calculate_mape(y_test, rf_predictions)
        xgb_mape = calculate_mape(y_test, xgb_predictions)
        
        print(f"LSTM/GRU MAPE: {lstm_gru_mape}")
        print(f"ARIMA MAPE: {arima_mape}")
        print(f"Random Forest MAPE: {rf_mape}")
        print(f"XGBoost MAPE: {xgb_mape}")
        
        # Prepare data for JSON output
        last_date = global_data['Date_reported'].iloc[-1]
        
        data = {
            'dates': global_data['Date_reported'].astype(str).tolist()[-test_size:],
            'actual': np.expm1(scaler.inverse_transform(y_test.reshape(-1, 1))).flatten().tolist(),
            'lstm_gru_predicted': np.expm1(scaler.inverse_transform(lstm_gru_predictions.reshape(-1, 1))).flatten().tolist(),
            'arima_predicted': np.expm1(scaler.inverse_transform(arima_predictions.reshape(-1, 1))).flatten().tolist(),
            'rf_predicted': np.expm1(scaler.inverse_transform(rf_predictions.reshape(-1, 1))).flatten().tolist(),
            'xgb_predicted': np.expm1(scaler.inverse_transform(xgb_predictions.reshape(-1, 1))).flatten().tolist(),
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