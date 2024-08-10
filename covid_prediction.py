import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import traceback
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    df = pd.read_csv(url)
    
    df_global = df.groupby('Date_reported')['New_cases'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    df_global['ds'] = pd.to_datetime(df_global['ds'])
    
    df_global['day_of_week'] = df_global['ds'].dt.dayofweek
    
    full_range = pd.date_range(start=df_global['ds'].min(), end=df_global['ds'].max(), freq='D')
    df_global = df_global.set_index('ds').reindex(full_range).reset_index().rename(columns={'index': 'ds'})
    df_global['y'] = df_global['y'].interpolate(method='linear')
    df_global['y'] = df_global['y'].clip(lower=0)  # Ensure non-negative values
    
    df_global['day_of_week'] = df_global['ds'].dt.dayofweek
    
    print(f"Data cleaned and interpolated. Shape: {df_global.shape}")
    return df_global

def train_and_predict_arima(df, future_days=30):
    print("Training ARIMA model and making predictions...")
    
    auto_model = auto_arima(df['y'], seasonal=True, m=7, stepwise=True, suppress_warnings=True, error_action="ignore", max_p=5, max_d=2, max_q=5)
    best_params = auto_model.get_params()
    
    model = ARIMA(df['y'], order=(best_params['order'][0], best_params['order'][1], best_params['order'][2]), seasonal_order=best_params['seasonal_order'])
    results = model.fit()
    
    forecast = results.forecast(steps=future_days)
    forecast = np.maximum(forecast, 0)  # Ensure non-negative predictions
    full_arima_predicted = np.maximum(np.concatenate([results.fittedvalues, forecast]), 0)
    
    print("ARIMA model training and prediction complete.")
    return full_arima_predicted, forecast

def prepare_data_lstm(data, look_back=7):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def train_and_predict_lstm(df, future_days=30):
    print("Training simplified LSTM model and making predictions...")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))
    
    look_back = 7
    X, y = prepare_data_lstm(scaled_data, look_back)
    
    model = Sequential([
        LSTM(30, activation='relu', input_shape=(look_back, 1)),  # Reduced units
        Dropout(0.1),  # Reduced dropout rate
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    last_sequence = scaled_data[-look_back:]
    predictions = []
    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, look_back, 1))
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    full_lstm_predicted = np.concatenate([df['y'].values, predictions.flatten()])
    
    print("LSTM model training and prediction complete.")
    return full_lstm_predicted, predictions.flatten()

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mae, rmse, mape

def update_historical_data(historical_data, new_data, date):
    historical_data[date] = {
        'arima': new_data['arima_future_predicted'][0],
        'lstm': new_data['lstm_future_predicted'][0]
    }
    
    with open('historical_predictions.json', 'w') as f:
        json.dump(historical_data, f)
    
    return historical_data

def main():
    try:
        print("Starting main function...")
        df = fetch_and_clean_data()
        last_date = df['ds'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        evaluation_days = 90
        
        full_arima_predicted, arima_predictions = train_and_predict_arima(df, future_days=30)
        full_lstm_predicted, lstm_predictions = train_and_predict_lstm(df, future_days=30)
        
        print("Preparing data for JSON...")
        data = {
            'dates': df['ds'].astype(str).tolist() + [d.strftime('%Y-%m-%d') for d in future_dates],
            'actual': df['y'].tolist() + [None] * 30,
            'arima_predicted': full_arima_predicted.tolist(),
            'arima_future_predicted': arima_predictions.tolist(),
            'lstm_predicted': full_lstm_predicted.tolist(),
            'lstm_future_predicted': lstm_predictions.tolist()
        }
        
        print("Calculating metrics...")
        arima_mae, arima_rmse, arima_mape = calculate_metrics(df['y'].values[-evaluation_days:], full_arima_predicted[-evaluation_days-30:-30])
        lstm_mae, lstm_rmse, lstm_mape = calculate_metrics(df['y'].values[-evaluation_days:], full_lstm_predicted[-evaluation_days-30:-30])
        
        data['arima_mae'] = float(arima_mae)
        data['arima_rmse'] = float(arima_rmse)
        data['arima_mape'] = float(arima_mape)
        data['lstm_mae'] = float(lstm_mae)
        data['lstm_rmse'] = float(lstm_rmse)
        data['lstm_mape'] = float(lstm_mape)
        data['last_updated'] = datetime.now().isoformat()
        
        print("Saving to JSON...")
        json_path = os.path.join(os.getcwd(), 'covid_predictions.json')
        print(f"Attempting to save JSON file at: {json_path}")

        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        if os.path.exists(json_path):
            print(f"JSON file saved successfully at: {json_path}")
        else:
            print("Failed to save JSON file.")
        
        # Update historical data
        try:
            with open('historical_predictions.json', 'r') as f:
                historical_data = json.load(f)
        except FileNotFoundError:
            historical_data = {}
        
        historical_data = update_historical_data(historical_data, data, last_date.strftime('%Y-%m-%d'))
        
        # Read back the file to ensure it was written correctly
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
            print(f"Read back JSON data: {saved_data}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
