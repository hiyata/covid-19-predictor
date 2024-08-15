import pandas as pd
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import json
import pickle
import matplotlib.pyplot as plt

def load_models_and_components():
    print("Loading models and components...")
    try:
        lstm_gru_model = tf.keras.models.load_model('lstm_model.h5')
        with open('quick_arima_model.pkl', 'rb') as f:
            arima_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('trial.json', 'r') as f:
            hyperparameters = json.load(f)['hyperparameters']['values']
        print("Models and components loaded successfully.")
        return lstm_gru_model, arima_model, scaler, hyperparameters
    except Exception as e:
        print(f"Error loading models and components: {str(e)}")
        return None, None, None, None

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    try:
        data = pd.read_csv(url)
        data['Date_reported'] = pd.to_datetime(data['Date_reported'])
        global_data = data.groupby('Date_reported').agg({'New_cases': 'sum'}).reset_index()
        print(f"Data cleaned. Shape: {global_data.shape}")
        return global_data
    except Exception as e:
        print(f"Error fetching or cleaning data: {str(e)}")
        return None

def prepare_data_for_prediction(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
    return np.array(X)

def make_predictions(lstm_gru_model, arima_model, scaler, data, sequence_length, num_days_to_predict):
    print("Making predictions...")
    X = prepare_data_for_prediction(data, sequence_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"Shape of X for LSTM/GRU prediction: {X.shape}")

    lstm_gru_predictions = []
    current_input = X[0]
    for _ in range(num_days_to_predict):
        prediction = lstm_gru_model.predict(current_input.reshape(1, sequence_length, 1))
        lstm_gru_predictions.append(prediction[0, 0])
        current_input = np.roll(current_input, -1)
        current_input[-1] = prediction[0, 0]

    lstm_gru_predictions = np.array(lstm_gru_predictions).reshape(-1, 1)
    lstm_gru_predictions = scaler.inverse_transform(lstm_gru_predictions)
    lstm_gru_predictions = np.expm1(lstm_gru_predictions)

    arima_predictions = arima_model.forecast(steps=num_days_to_predict)
    arima_predictions = scaler.inverse_transform(arima_predictions.reshape(-1, 1))
    arima_predictions = np.expm1(arima_predictions)

    print("Predictions completed.")
    return lstm_gru_predictions, arima_predictions

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    return mae, mape

def plot_predictions(historical_dates, future_dates, original_cases, lstm_gru_predictions, arima_predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(historical_dates, original_cases, label='Historical Cases', marker='o')
    plt.plot(future_dates, lstm_gru_predictions, label='LSTM/GRU Predicted New Cases', marker='o')
    plt.plot(future_dates, arima_predictions, label='ARIMA Predicted New Cases', marker='o')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.title('Historical Cases and Predictions')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('new_prediction_comparison.png')
    plt.close()
    print("Prediction graph saved as 'new_prediction_comparison.png'")

def main():
    print("Starting COVID-19 prediction script...")
    
    lstm_gru_model, arima_model, scaler, hyperparameters = load_models_and_components()
    if lstm_gru_model is None or arima_model is None or scaler is None or hyperparameters is None:
        print("Error: Models or components could not be loaded. Exiting...")
        return

    sequence_length = hyperparameters['sequence_length']
    print(f"Sequence length: {sequence_length}")

    global_data = fetch_and_clean_data()
    if global_data is None:
        print("Error: Data could not be fetched or cleaned. Exiting...")
        return

    original_cases = global_data['New_cases'].values[-30:]
    global_data['New_cases'] = np.log1p(global_data['New_cases'])
    global_data['New_cases'] = scaler.transform(global_data['New_cases'].values.reshape(-1, 1))

    latest_data = global_data['New_cases'].values[-sequence_length:]
    print(f"Shape of latest_data: {latest_data.shape}")

    num_days_to_predict = 7
    lstm_gru_predictions, arima_predictions = make_predictions(lstm_gru_model, arima_model, scaler, latest_data, sequence_length, num_days_to_predict)

    last_date = global_data['Date_reported'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days_to_predict)

    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM_GRU_Predicted': lstm_gru_predictions.flatten(),
        'ARIMA_Predicted': arima_predictions.flatten()
    })

    print("\nPredictions for the next 7 days:")
    print(prediction_df.to_string(index=False))

    historical_dates = global_data['Date_reported'].iloc[-30:]
    
    plot_predictions(historical_dates, future_dates, original_cases, 
                     lstm_gru_predictions.flatten(), arima_predictions.flatten())

    # Calculate and print MAPE and MAE for the last 7 days of actual data
    actual_last_7_days = original_cases[-7:]
    lstm_gru_mae, lstm_gru_mape = calculate_metrics(actual_last_7_days, lstm_gru_predictions.flatten())
    arima_mae, arima_mape = calculate_metrics(actual_last_7_days, arima_predictions.flatten())

    print("\nError Metrics for the Last 7 Days:")
    print(f"LSTM/GRU - MAE: {lstm_gru_mae:.2f}, MAPE: {lstm_gru_mape:.2f}")
    print(f"ARIMA - MAE: {arima_mae:.2f}, MAPE: {arima_mape:.2f}")

    print("Prediction script completed.")

if __name__ == "__main__":
    main()