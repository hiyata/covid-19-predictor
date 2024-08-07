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

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    df = pd.read_csv(url)
    
    # Aggregate global daily cases
    df_global = df.groupby('Date_reported')['New_cases'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    df_global['ds'] = pd.to_datetime(df_global['ds'])
    
    # Add day of the week as a feature
    df_global['day_of_week'] = df_global['ds'].dt.dayofweek
    
    # Fill missing dates and interpolate values
    full_range = pd.date_range(start=df_global['ds'].min(), end=df_global['ds'].max(), freq='D')
    df_global = df_global.set_index('ds').reindex(full_range).reset_index().rename(columns={'index': 'ds'})
    df_global['y'] = df_global['y'].interpolate(method='linear')
    
    # Add day of the week for the filled dates
    df_global['day_of_week'] = df_global['ds'].dt.dayofweek
    
    print(f"Data cleaned and interpolated. Shape: {df_global.shape}")
    return df_global

def prepare_data(df, look_back=15):
    data = df['y'].values
    day_of_week = df['day_of_week'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Apply log transformation to stabilize variance
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

def train_and_predict_hybrid(df, look_back=15):
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
    
    # Make predictions for known dates
    known_predictions = model.predict(X)
    known_predictions = scaler.inverse_transform(known_predictions).flatten()
    known_predictions = np.expm1(known_predictions)
    
    # Prepare data for future predictions
    future_X = X[-1:]
    future_day_of_week = df['day_of_week'].values[-look_back:].reshape(-1, 1)
    
    future_predictions = []
    for i in range(30):
        pred = model.predict(future_X)
        future_predictions.append(pred[0, 0])
        next_day_of_week = (future_day_of_week[-1] + 1) % 7
        pred_reshaped = np.hstack((pred.reshape(1, 1), next_day_of_week.reshape(1, 1)))
        future_X = np.concatenate((future_X[:, 1:, :], pred_reshaped.reshape(1, 1, 2)), axis=1)
        future_day_of_week = np.append(future_day_of_week[1:], next_day_of_week)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_predictions = np.expm1(future_predictions).flatten()
    
    print("Hybrid model training and prediction complete.")
    return known_predictions, future_predictions

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def main():
    try:
        print("Starting main function...")
        df = fetch_and_clean_data()
        known_predictions, future_predictions = train_and_predict_hybrid(df)
        
        # Separate actual and predicted data
        actual = df['y'].values
        predicted = known_predictions[:len(actual)]  # Predictions for known dates
        
        print("Preparing data for JSON...")
        data = {
            'dates': df['ds'].astype(str).tolist(),
            'actual': actual.tolist(),
            'predicted': predicted.tolist(),
            'future_dates': [str(df['ds'].iloc[-1] + timedelta(days=i)) for i in range(1, 31)],
            'future_predicted': future_predictions.tolist()
        }
        
        print("Calculating metrics...")
        mae, rmse, mape = calculate_metrics(actual[-30:], predicted[-30:])  # Use last 30 days for metrics
        
        data['mae'] = float(mae)
        data['rmse'] = float(rmse)
        data['mape'] = float(mape)
        data['last_updated'] = datetime.now().isoformat()
        
        print("Saving to JSON...")
        json_path = os.path.join(os.getcwd(), 'covid_predictions.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        print(f"JSON file saved successfully at: {json_path}")
        
        # Print some debug information
        print(f"Number of actual data points: {len(data['actual'])}")
        print(f"Number of predicted data points: {len(data['predicted'])}")
        print(f"First few actual values: {data['actual'][:5]}")
        print(f"First few predicted values: {data['predicted'][:5]}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()