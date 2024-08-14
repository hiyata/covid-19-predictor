import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
from statsmodels.tsa.arima.model import ARIMA

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
    
    print(f"Data cleaned. Shape: {df_global.shape}")
    return df_global

def load_lstm_model():
    print("Loading LSTM model...")
    with open('build_config.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('checkpoint.weights.h5')
    return model

def load_arima_model():
    print("Loading ARIMA model...")
    with open('quick_arima_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def prepare_data_lstm(data, look_back=7):
    return data[-look_back:].reshape(1, look_back, 1)

def predict_lstm(model, data, scaler, future_days=7):
    print("Making LSTM predictions...")
    last_sequence = prepare_data_lstm(data)
    predictions = []
    for _ in range(future_days):
        next_pred = model.predict(last_sequence)
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred[0, 0]
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def predict_arima(model, steps=7):
    print("Making ARIMA predictions...")
    return model.forecast(steps=steps)

def update_predictions(existing_predictions, new_predictions, dates):
    existing_predictions['actual'][-1] = new_predictions['actual'][0]
    existing_predictions['dates'].extend(new_predictions['dates'][1:])
    existing_predictions['actual'].extend(new_predictions['actual'][1:])
    existing_predictions['lstm_predicted'].extend(new_predictions['lstm_predicted'][1:])
    existing_predictions['arima_predicted'].extend(new_predictions['arima_predicted'][1:])
    return existing_predictions

def save_predictions(predictions, filename):
    with open(filename, 'w') as f:
        json.dump(predictions, f)

def load_predictions(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main():
    try:
        print("Starting main function...")
        df = fetch_and_clean_data()
        if df is None:
            print("Failed to fetch data. Exiting.")
            return

        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        # Load models
        lstm_model = load_lstm_model()
        arima_model = load_arima_model()
        
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['New_cases'].values.reshape(-1, 1))
        
        # Make predictions
        lstm_predictions = predict_lstm(lstm_model, scaled_data, scaler, future_days=7)
        arima_predictions = predict_arima(arima_model, steps=7)
        
        new_predictions = {
            'dates': [last_date.strftime('%Y-%m-%d')] + [d.strftime('%Y-%m-%d') for d in future_dates],
            'actual': [df['New_cases'].iloc[-1]] + [None] * 7,
            'lstm_predicted': lstm_predictions.tolist(),
            'arima_predicted': arima_predictions.tolist()
        }
        
        # Load existing predictions or create new if not exists
        existing_predictions = load_predictions('covid_predictions.json')
        if existing_predictions:
            updated_predictions = update_predictions(existing_predictions, new_predictions, future_dates)
        else:
            updated_predictions = new_predictions
        
        # Save updated predictions
        save_predictions(updated_predictions, 'covid_predictions.json')
        
        # Save this prediction for evaluation
        evaluation_filename = f'prediction_{last_date.strftime("%Y%m%d")}.json'
        save_predictions(new_predictions, evaluation_filename)
        
        print("Predictions updated and saved successfully.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()