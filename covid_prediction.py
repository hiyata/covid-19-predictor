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
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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
    
    # Convert 'Date_reported' to datetime
    df['Date_reported'] = pd.to_datetime(df['Date_reported'])
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date_reported')
    
    # Aggregate global daily cases
    df_global = df.groupby('Date_reported')['New_cases'].sum().reset_index()
    
    # Handle missing values
    df_global['New_cases'] = df_global['New_cases'].fillna(0)
    
    # Check for negative values
    negative_cases = df_global[df_global['New_cases'] < 0]
    if not negative_cases.empty:
        print(f"Warning: Found {len(negative_cases)} days with negative case counts. These will be set to 0.")
        print(negative_cases)
        df_global['New_cases'] = df_global['New_cases'].clip(lower=0)
    
    # Check for and handle zero values
    zero_cases = df_global[df_global['New_cases'] == 0]
    if not zero_cases.empty:
        print(f"Note: Found {len(zero_cases)} days with zero case counts.")
    
    # Add a small constant to avoid log(0)
    epsilon = 1e-1  # This value can be adjusted based on your data
    df_global['New_cases'] = df_global['New_cases'] + epsilon
    
    # Apply log transformation
    df_global['New_cases_log'] = np.log1p(df_global['New_cases'])
    
    # Check for infinity or NaN values after transformation
    inf_or_nan = df_global[~np.isfinite(df_global['New_cases_log'])]
    if not inf_or_nan.empty:
        print(f"Warning: Found {len(inf_or_nan)} infinite or NaN values after log transformation.")
        print(inf_or_nan)
        # Replace inf or NaN with the mean of finite values
        mean_cases = df_global['New_cases_log'][np.isfinite(df_global['New_cases_log'])].mean()
        df_global['New_cases_log'] = df_global['New_cases_log'].replace([np.inf, -np.inf, np.nan], mean_cases)
    
    print(f"Data cleaned and transformed. Shape: {df_global.shape}")
    print(f"Date range: {df_global['Date_reported'].min()} to {df_global['Date_reported'].max()}")
    print(f"New_cases range: {df_global['New_cases'].min()} to {df_global['New_cases'].max()}")
    print(f"New_cases_log range: {df_global['New_cases_log'].min()} to {df_global['New_cases_log'].max()}")
    
    return df_global

def prepare_data(data, sequence_length=90):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['New_cases'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    X_flat = X.reshape(X.shape[0], -1)  # Flatten for non-sequential models
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM/GRU
    
    return X, X_flat, y, scaler

def build_lstm_gru_model(sequence_length):
    model = Sequential()
    model.add(Input(shape=(sequence_length, 1)))
    model.add(GRU(units=40, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(units=40))
    model.add(Dense(units=100))
    model.add(Dense(units=70))
    model.add(Dense(units=1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007919746988842461),
                  loss='mean_absolute_percentage_error')
    
    return model

def arima_model(train_data, order=(5,1,0)):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def xgboost_model(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def main():
    try:
        print("Starting main function...")
        global_data = fetch_and_clean_data()
        
        sequence_length = 60
        X, X_flat, y, scaler = prepare_data(global_data, sequence_length)
        
        # Use the last 7 days for final evaluation
        train_data = global_data['New_cases'].values[:-7]
        X_train, X_flat_train, y_train = X[:-7], X_flat[:-7], y[:-7]
        X_test, X_flat_test = X[-7:], X_flat[-7:]
        
        # LSTM/GRU model
        print("Training LSTM/GRU model...")
        lstm_gru_model = build_lstm_gru_model(sequence_length)
        lstm_gru_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)
        lstm_gru_predictions = lstm_gru_model.predict(X_test)
        
        # ARIMA model
        print("Training ARIMA model...")
        arima_model_fit = arima_model(train_data)
        arima_predictions = arima_model_fit.forecast(steps=7)
        
        # Random Forest model
        print("Training Random Forest model...")
        rf_model = random_forest_model(X_flat_train, y_train)
        rf_predictions = rf_model.predict(X_flat_test)
        
        # XGBoost model
        print("Training XGBoost model...")
        xgb_model = xgboost_model(X_flat_train, y_train)
        xgb_predictions = xgb_model.predict(X_flat_test)
        
        # Inverse transform predictions and actual values
        actual_cases = np.expm1(global_data['New_cases'].values[-7:])
        lstm_gru_predictions = np.expm1(scaler.inverse_transform(lstm_gru_predictions).flatten())
        arima_predictions = np.expm1(arima_predictions)
        rf_predictions = np.expm1(scaler.inverse_transform(rf_predictions.reshape(-1, 1)).flatten())
        xgb_predictions = np.expm1(scaler.inverse_transform(xgb_predictions.reshape(-1, 1)).flatten())
        
        # Calculate MAPE for the final predictions
        lstm_gru_mape = calculate_mape(actual_cases, lstm_gru_predictions)
        arima_mape = calculate_mape(actual_cases, arima_predictions)
        rf_mape = calculate_mape(actual_cases, rf_predictions)
        xgb_mape = calculate_mape(actual_cases, xgb_predictions)
        
        # Print the final MAPE values
        print("\nFinal MAPE values:")
        print(f"LSTM/GRU Final MAPE: {lstm_gru_mape}")
        print(f"ARIMA Final MAPE: {arima_mape}")
        print(f"Random Forest Final MAPE: {rf_mape}")
        print(f"XGBoost Final MAPE: {xgb_mape}")
        
        # Create a DataFrame to compare actual vs predicted
        comparison_df = pd.DataFrame({
            'Date': global_data['Date_reported'].values[-7:],
            'Actual': actual_cases,
            'LSTM_GRU_Predicted': lstm_gru_predictions,
            'ARIMA_Predicted': arima_predictions,
            'Random_Forest_Predicted': rf_predictions,
            'XGBoost_Predicted': xgb_predictions
        })
        
        # Display the comparison DataFrame
        print("\nComparison of actual vs predicted cases for the last 7 days:")
        print(comparison_df)
        
        # Save results to JSON
        results = {
            'dates': comparison_df['Date'].astype(str).tolist(),
            'actual': comparison_df['Actual'].tolist(),
            'lstm_gru_predictions': comparison_df['LSTM_GRU_Predicted'].tolist(),
            'arima_predictions': comparison_df['ARIMA_Predicted'].tolist(),
            'rf_predictions': comparison_df['Random_Forest_Predicted'].tolist(),
            'xgb_predictions': comparison_df['XGBoost_Predicted'].tolist(),
            'mape': {
                'lstm_gru': lstm_gru_mape,
                'arima': arima_mape,
                'random_forest': rf_mape,
                'xgboost': xgb_mape
            },
            'last_updated': datetime.now().isoformat()
        }
        
        json_path = os.path.join(os.getcwd(), 'covid_predictions.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)
        
        print(f"\nResults saved to: {json_path}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(comparison_df['Date'], comparison_df['Actual'], label='Actual', marker='o')
        plt.plot(comparison_df['Date'], comparison_df['LSTM_GRU_Predicted'], label='LSTM/GRU', marker='s')
        plt.plot(comparison_df['Date'], comparison_df['ARIMA_Predicted'], label='ARIMA', marker='^')
        plt.plot(comparison_df['Date'], comparison_df['Random_Forest_Predicted'], label='Random Forest', marker='D')
        plt.plot(comparison_df['Date'], comparison_df['XGBoost_Predicted'], label='XGBoost', marker='v')
        plt.title('COVID-19 Case Predictions for Last 7 Days')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('covid_predictions_plot.png')
        print("Plot saved as covid_predictions_plot.png")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()