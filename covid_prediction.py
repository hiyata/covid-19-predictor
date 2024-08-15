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
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

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

def prepare_data(df, sequence_length=90):
    data = df['y'].values
    day_of_week = df['day_of_week'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Apply log transformation to stabilize variance
    data = np.log1p(data)
    data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(np.hstack((data[i:i+sequence_length], day_of_week[i:i+sequence_length])))
        y.append(data[i+sequence_length, 0])
    
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def create_model(input_shape):
    model = Sequential()
    
    # Layer 0: GRU
    model.add(GRU(40, input_shape=input_shape, return_sequences=True))
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
    
    return model

def train_and_predict(df, sequence_length=90):
    print("Training model and making predictions...")
    X, y, scaler = prepare_data(df, sequence_length)
    
    model = create_model((X.shape[1], X.shape[2]))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007919746988842461)
    model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer)
    
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    model.fit(X, y, epochs=60, batch_size=32, verbose=2, callbacks=[early_stopping])
    
    # Make predictions for known dates
    known_predictions = model.predict(X)
    known_predictions = scaler.inverse_transform(known_predictions).flatten()
    known_predictions = np.expm1(known_predictions)
    
    # Prepare data for future predictions
    last_sequence = X[-1:]
    
    future_predictions = []
    for i in range(67):  # 67 days prediction (60 for comparison + 7 for future)
        pred = model.predict(last_sequence)
        future_predictions.append(pred[0, 0])
        next_day_of_week = (last_sequence[0, -1, 1] + 1) % 7
        new_data_point = np.hstack((pred, [[next_day_of_week]]))
        last_sequence = np.concatenate((last_sequence[:, 1:, :], new_data_point.reshape(1, 1, 2)), axis=1)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_predictions = np.expm1(future_predictions).flatten()
    
    print("Model training and prediction complete.")
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
        known_predictions, future_predictions = train_and_predict(df)
        
        # Separate actual and predicted data
        actual = df['y'].values
        predicted = known_predictions[:len(actual)]  # Predictions for known dates
        
        print("Preparing data for JSON...")
        last_date = df['ds'].iloc[-1]
        sixty_days_ago = last_date - timedelta(days=60)
        
        data = {
            'dates': df['ds'].astype(str).tolist(),
            'actual': actual.tolist(),
            'predicted': predicted.tolist(),
            'comparison_dates': [str(sixty_days_ago + timedelta(days=i)) for i in range(67)],
            'comparison_actual': actual[-60:].tolist() + [None] * 7,  # Last 60 known + 7 unknown
            'comparison_predicted': future_predictions.tolist(),
            'future_dates': [str(last_date + timedelta(days=i)) for i in range(1, 8)],
            'future_predicted': future_predictions[-7:].tolist()
        }
        
        print("Calculating metrics...")
        mae, rmse, mape = calculate_metrics(actual[-60:], future_predictions[:60])  # Use last 60 days for metrics
        
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
        print(f"Number of comparison data points: {len(data['comparison_predicted'])}")
        print(f"Number of future prediction data points: {len(data['future_predicted'])}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()