import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
import os

# Function to fetch and preprocess data
def fetch_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(url)
    df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%m/%d/%y')  # Specify the format
    df_global = df_melted.groupby('Date')['Confirmed'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    return df_global

# Function to train model and make predictions
def train_and_predict(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Function to calculate performance metrics
def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual))
    return mae, rmse, mape

# Main function
def main():
    df = fetch_data()
    forecast = train_and_predict(df)
    
    # Prepare data for JSON
    data = {
        'dates': df['ds'].astype(str).tolist() + forecast['ds'].tail(30).astype(str).tolist(),
        'actual': df['y'].tolist() + [None] * 30,
        'predicted': forecast['yhat'].tolist(),
    }
    
    # Calculate metrics
    actual = df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    mae, rmse, mape = calculate_metrics(actual, predicted)
    
    data['mae'] = mae
    data['rmse'] = rmse
    data['mape'] = mape
    data['last_updated'] = datetime.now().isoformat()
    
    # Save to JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'covid_predictions.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"JSON file saved at: {json_path}")

if __name__ == "__main__":
    main()