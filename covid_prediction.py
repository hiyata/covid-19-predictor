import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import requests
import json
import os

def fetch_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(url)
    df_melted = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')
    df_melted['Date'] = pd.to_datetime(df_melted['Date'])
    df_global = df_melted.groupby('Date')['Confirmed'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    return df_global

def train_and_predict(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual))
    return mae, rmse, mape

def main():
    df = fetch_data()
    forecast = train_and_predict(df)
    
    data = {
        'dates': df['ds'].astype(str).tolist() + forecast['ds'].tail(30).astype(str).tolist(),
        'actual': df['y'].tolist() + [None] * 30,
        'predicted': forecast['yhat'].tolist(),
    }
    
    actual = df['y'].values
    predicted = forecast['yhat'][:len(actual)].values
    mae, rmse, mape = calculate_metrics(actual, predicted)
    
    data['mae'] = mae
    data['rmse'] = rmse
    data['mape'] = mape
    data['last_updated'] = datetime.now().isoformat()
    
    os.makedirs('assets/data', exist_ok=True)
    with open('assets/data/covid_predictions.json', 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()