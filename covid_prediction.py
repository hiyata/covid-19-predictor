import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json
import os
import sys
import traceback

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df_global = df.groupby('date')['new_cases'].sum().reset_index()
    df_global.columns = ['ds', 'y']
    df_global['ds'] = pd.to_datetime(df_global['ds'])
    
    # Fill missing dates and interpolate values
    full_range = pd.date_range(start=df_global['ds'].min(), end=df_global['ds'].max(), freq='D')
    df_global = df_global.set_index('ds').reindex(full_range).reset_index().rename(columns={'index': 'ds'})
    df_global['y'] = df_global['y'].interpolate(method='linear')

    print(f"Data cleaned and interpolated. Shape: {df_global.shape}")
    return df_global

def train_and_predict(df):
    print("Training model and making predictions...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=30)  # 30 days forecast
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
        df = fetch_and_clean_data()
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
        json_path = os.path.join(os.getcwd(), 'covid_predictions.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        print(f"JSON file saved at: {json_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
