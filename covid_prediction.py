import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
    df = pd.read_csv(url)
    df_global = df.groupby('Date_reported')['New_cases'].sum().reset_index()
    df_global['Date_reported'] = pd.to_datetime(df_global['Date_reported'])
    df_global.set_index('Date_reported', inplace=True)
    
    # Apply log transformation
    df_global['New_cases'] = np.log1p(df_global['New_cases'])
    
    print(f"Data cleaned. Shape: {df_global.shape}")
    return df_global

def load_lstm_model():
    print("Loading LSTM model...")
    try:
        model = load_model('lstm_model.h5')
        print("LSTM model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading LSTM model: {str(e)}")
        return None

def load_arima_model():
    print("Loading ARIMA model...")
    try:
        with open('arima_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading ARIMA model: {str(e)}")
        return None

def load_scaler():
    print("Loading scaler...")
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

def load_hyperparameters():
    print("Loading hyperparameters...")
    try:
        with open('hyperparameters.json', 'r') as f:
            hyperparameters = json.load(f)
        return hyperparameters
    except Exception as e:
        print(f"Error loading hyperparameters: {str(e)}")
        return None

def prepare_data(data, sequence_length, scaler):
    scaled_data = scaler.transform(data.values.reshape(-1, 1))
    
    X = []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X

def predict_lstm(model, data, scaler, sequence_length, future_days=7):
    last_sequence = data[-sequence_length:]
    predicted = []
    
    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        predicted.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    predicted = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
    return np.expm1(predicted.flatten())  # Reverse log transformation

def predict_arima(model, future_days=7):
    forecast = model.forecast(steps=future_days)
    return np.expm1(forecast)  # Reverse log transformation

def save_predictions(predictions, filename='covid_predictions.json'):
    with open(filename, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {filename}")

def load_predictions(filename='covid_predictions.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def plot_predictions(dates, actual, lstm_pred, arima_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', marker='o')
    plt.plot(dates, lstm_pred, label='LSTM Prediction', marker='s')
    plt.plot(dates, arima_pred, label='ARIMA Prediction', marker='^')
    plt.title('COVID-19 New Cases: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_plot.png')
    plt.close()
    print("Prediction plot saved as 'prediction_plot.png'")

def main():
    df = fetch_and_clean_data()
    if df is None:
        print("Failed to fetch data. Exiting.")
        return

    lstm_model = load_lstm_model()
    arima_model = load_arima_model()
    scaler = load_scaler()
    hyperparameters = load_hyperparameters()

    if lstm_model is None or arima_model is None or scaler is None or hyperparameters is None:
        print("Failed to load models, scaler, or hyperparameters. Exiting.")
        return

    sequence_length = hyperparameters['hyperparameters']['values']['sequence_length']

    X = prepare_data(df['New_cases'], sequence_length, scaler)

    lstm_predictions = predict_lstm(lstm_model, X[-1], scaler, sequence_length)
    arima_predictions = predict_arima(arima_model)

    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    
    new_predictions = {
        'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'lstm_predicted': lstm_predictions.tolist(),
        'arima_predicted': arima_predictions.tolist(),
        'last_updated': datetime.now().isoformat()
    }

    existing_predictions = load_predictions()
    if existing_predictions:
        existing_predictions['dates'] = existing_predictions['dates'][-6:] + new_predictions['dates'][-1:]
        existing_predictions['lstm_predicted'] = existing_predictions['lstm_predicted'][-6:] + new_predictions['lstm_predicted'][-1:]
        existing_predictions['arima_predicted'] = existing_predictions['arima_predicted'][-6:] + new_predictions['arima_predicted'][-1:]
        existing_predictions['last_updated'] = new_predictions['last_updated']
    else:
        existing_predictions = new_predictions

    save_predictions(existing_predictions)

    print("\n7-day forecast:")
    for date, lstm, arima in zip(existing_predictions['dates'], 
                                 existing_predictions['lstm_predicted'], 
                                 existing_predictions['arima_predicted']):
        print(f"Date: {date}, LSTM: {lstm:.2f}, ARIMA: {arima:.2f}")

    # Plot the predictions
    plot_dates = pd.to_datetime(existing_predictions['dates'])
    actual_cases = np.expm1(df['New_cases'].iloc[-7:].values)
    plot_predictions(plot_dates, actual_cases, 
                     existing_predictions['lstm_predicted'], 
                     existing_predictions['arima_predicted'])

if __name__ == "__main__":
    main()