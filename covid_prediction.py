import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
import os
import sys
import traceback

def fetch_data():
    print("Fetching data...")
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df_global = df.groupby('date')['new_cases'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    df_global['ds'] = pd.to_datetime(df_global['ds'], format='%Y-%m-%d')
    print(f"Data fetched. Shape: {df_global.shape}")
    return df_global

def train_and_predict(df):
    print("Training model and making predictions...")
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    print("Model training and prediction complete.")
    return forecast

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def main():
    try:
        print("Starting main function...")
        df = fetch_data()
        forecast = train_and_predict(df)
        
        print("Preparing data for JSON...")
        data = {
            'dates': df['ds'].astype(str).tolist() + forecast['ds'].tail(30).astype(str).tolist(),
            'actual': df['y'].tolist() + [None] * 30,
            'predicted': forecast['yhat'].tolist(),
        }
        
        print("Calculating metrics...")
        actual = df['y'].values
        predicted = forecast['yhat'][:len(actual)].values
        mae, rmse, mape = calculate_metrics(actual, predicted)
        
        data['mae'] = float(mae)
        data['rmse'] = float(rmse)
        data['mape'] = float(mape)
        data['last_updated'] = datetime.now().isoformat()
        
        print("Saving to JSON...")
        with open('covid_predictions.json', 'w') as f:
            json.dump(data, f)
        
        print("JSON file saved successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()