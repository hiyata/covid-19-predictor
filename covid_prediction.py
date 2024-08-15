import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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
    
    # Apply log transformation to the 'New_cases' column
    df_global['New_cases'] = np.log1p(df_global['New_cases'])
    
    print(f"Data cleaned. Shape: {df_global.shape}")
    return df_global

def load_lstm_model():
    print("Loading LSTM model...")
    try:
        # Load architecture
        with open('lstm_model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        
        # Load weights
        model.load_weights("lstm_model.weights.h5")
        
        print("LSTM model loaded successfully.")
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading LSTM model: {str(e)}")
        return None

def load_arima_model():
    print("Loading ARIMA model...")
    try:
        with open('quick_arima_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("ARIMA model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading ARIMA model: {str(e)}")
        return None

def load_scaler():
    print("Loading scaler...")
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {str(e)}")
        return None

def prepare_data(data, sequence_length, scaler):
    print(f"Preparing data with sequence length: {sequence_length}")
    scaled_data = scaler.transform(data.values.reshape(-1, 1))
    X = []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

def predict_with_models(data, sequence_length):
    lstm_model = load_lstm_model()
    arima_model = load_arima_model()
    scaler = load_scaler()
    
    if lstm_model is None or arima_model is None or scaler is None:
        print("Error: Models or scaler could not be loaded. Exiting...")
        return
    
    X = prepare_data(data, sequence_length, scaler)
    lstm_predictions = lstm_model.predict(X)
    arima_predictions = arima_model.predict(start=len(data), end=len(data) + len(X) - 1)
    
    # Inverse transform to get the original scale
    lstm_predictions = np.expm1(scaler.inverse_transform(lstm_predictions).flatten())
    arima_predictions = np.expm1(scaler.inverse_transform(arima_predictions.reshape(-1, 1)).flatten())
    
    print("Predictions complete. Returning results...")
    return lstm_predictions, arima_predictions

def main():
    print("Starting prediction script...")
    data = fetch_and_clean_data()
    if data is None:
        print("Error: Data could not be fetched or cleaned. Exiting...")
        return
    
    sequence_length = 60  # This should match the sequence length used during training
    lstm_predictions, arima_predictions = predict_with_models(data, sequence_length)
    
    # Display or save the predictions as needed
    prediction_dates = data.index[-len(lstm_predictions):]
    comparison_df = pd.DataFrame({
        'Date': prediction_dates,
        'LSTM_GRU_Predicted': lstm_predictions,
        'ARIMA_Predicted': arima_predictions
    })
    
    print("\nComparison of LSTM/GRU vs ARIMA Predicted New Cases:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()
