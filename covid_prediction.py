import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
import os
import sys

# Function to fetch and preprocess data
def fetch_data():
    print("Fetching data...")
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(url)
    df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%m/%d/%y')
    df_global = df_melted.groupby('Date')['Confirmed'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    print(f"Data fetched. Shape: {df_global.shape}")
    return df_global

# Function to train model and make predictions
def train_and_predict(df):
    print("Training model and making predictions...")
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    print("Model training and prediction complete.")
    return forecast

# Function to calculate performance metrics
def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual))
    return mae, rmse, mape

# Main function
def main():
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
    
    data['mae'] = mae
    data['rmse'] = rmse
    data['mape'] = mape
    data['last_updated'] = datetime.now().isoformat()
    
    print("Saving to JSON...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'covid_predictions.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"JSON file saved at: {json_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")

if __name__ == "__main__":
    main()