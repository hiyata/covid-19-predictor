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
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

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

def prepare_data(df, look_back=15):
    data = df['y'].values
    day_of_week = df['day_of_week'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    data = np.log1p(data)
    data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        combined_features = np.hstack((data[i:(i + look_back)], day_of_week[i:(i + look_back)]))
        X.append(combined_features)
        y.append(data[i + look_back, 0])
    
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def train_and_predict_hybrid(df, look_back=15, future_days=30):
    print("Training hybrid model and making predictions...")
    X, y, scaler = prepare_data(df, look_back)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, X.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(tf.keras.layers.Reshape((50, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(GRU(50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model.fit(X, y, epochs=50, batch_size=32, verbose=2, callbacks=[early_stopping])
    
    future_X = X[-1:]
    future_day_of_week = df['day_of_week'].values[-look_back:].reshape(-1, 1)
    
    predictions = []
    for i in range(future_days):
        pred = model.predict(future_X)
        predictions.append(pred[0, 0])
        next_day_of_week = (future_day_of_week[-1] + 1) % 7
        pred_reshaped = np.hstack((pred.reshape(1, 1), next_day_of_week.reshape(1, 1)))
        future_X = np.concatenate((future_X[:, 1:, :], pred_reshaped.reshape(1, 1, 2)), axis=1)
        future_day_of_week = np.append(future_day_of_week[1:], next_day_of_week)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative predictions
    
    full_predicted = np.concatenate((df['y'].values, predictions.flatten()))
    
    print("Hybrid model training and prediction complete.")
    return full_predicted, predictions

def train_and_predict_prophet(df, future_days=30):
    print("Training Prophet model and making predictions...")
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True,
                    changepoint_prior_scale=0.5, seasonality_prior_scale=10)
    model.add_country_holidays(country_name='US')
    model.fit(df[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=future_days)
    forecast = model.predict(future)
    
    prophet_predictions = np.maximum(forecast['yhat'].values[-future_days:], 0)
    full_prophet_predicted = np.maximum(forecast['yhat'].values, 0)
    
    print("Prophet model training and prediction complete.")
    return full_prophet_predicted, prophet_predictions

def train_and_predict_arima(df, future_days=30):
    print("Training ARIMA model and making predictions...")
    
    # Automatically find the best ARIMA parameters
    auto_model = auto_arima(df['y'], seasonal=True, m=7, stepwise=True, suppress_warnings=True, 
                            error_action="ignore", max_p=5, max_d=2, max_q=5)
    best_params = auto_model.get_params()
    
    # Train the ARIMA model with the best parameters
    model = ARIMA(df['y'], order=best_params['order'], seasonal_order=best_params['seasonal_order'])
    results = model.fit()
    
    # Make predictions
    forecast = results.forecast(steps=future_days)
    forecast = np.maximum(forecast, 0)  # Ensure non-negative predictions
    full_arima_predicted = np.maximum(np.concatenate([results.fittedvalues, forecast]), 0)
    
    print("ARIMA model training and prediction complete.")
    return full_arima_predicted, forecast

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mae, rmse, mape

def main():
    try:
        print("Starting main function...")
        df = fetch_and_clean_data()
        last_date = df['ds'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        evaluation_days = 30  # Use the last 30 days for evaluation
        
        full_predicted, future_predictions = train_and_predict_hybrid(df, future_days=30)
        full_prophet_predicted, prophet_predictions = train_and_predict_prophet(df, future_days=30)
        full_arima_predicted, arima_predictions = train_and_predict_arima(df, future_days=30)
        
        print("Preparing data for JSON...")
        data = {
            'dates': df['ds'].astype(str).tolist() + [d.strftime('%Y-%m-%d') for d in future_dates],
            'actual': df['y'].tolist() + [None] * 30,
            'predicted': full_predicted.tolist(),
            'future_predicted': future_predictions.flatten().tolist(),
            'prophet_predicted': full_prophet_predicted.tolist(),
            'prophet_future_predicted': prophet_predictions.tolist(),
            'arima_predicted': full_arima_predicted.tolist(),
            'arima_future_predicted': arima_predictions.tolist()
        }
        
        print("Calculating metrics...")
        hybrid_mae, hybrid_rmse, hybrid_mape = calculate_metrics(df['y'].values[-evaluation_days:], full_predicted[-evaluation_days-30:-30])
        prophet_mae, prophet_rmse, prophet_mape = calculate_metrics(df['y'].values[-evaluation_days:], full_prophet_predicted[-evaluation_days-30:-30])
        arima_mae, arima_rmse, arima_mape = calculate_metrics(df['y'].values[-evaluation_days:], full_arima_predicted[-evaluation_days-30:-30])
        
        data['hybrid_mae'] = float(hybrid_mae)
        data['hybrid_rmse'] = float(hybrid_rmse)
        data['hybrid_mape'] = float(hybrid_mape)
        data['prophet_mae'] = float(prophet_mae)
        data['prophet_rmse'] = float(prophet_rmse)
        data['prophet_mape'] = float(prophet_mape)
        data['arima_mae'] = float(arima_mae)
        data['arima_rmse'] = float(arima_rmse)
        data['arima_mape'] = float(arima_mape)
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
        
        historical_data[last_date.strftime('%Y-%m-%d')] = {
            'hybrid': future_predictions[0],
            'prophet': prophet_predictions[0],
            'arima': arima_predictions[0]
        }
        
        with open('historical_predictions.json', 'w') as f:
            json.dump(historical_data, f)
        
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