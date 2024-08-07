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
    
    # Remove rows where 'y' is 0 or NaN
    df_global = df_global[(df_global['y'] != 0) & (df_global['y'].notna())]
    
    # Resample to weekly data
    df_global = df_global.set_index('ds')
    df_weekly = df_global.resample('W-MON').sum().reset_index()
    
    print(f"Data cleaned and resampled. Shape: {df_weekly.shape}")
    return df_weekly

def train_and_predict(df):
    print("Training model and making predictions...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=8, freq='W')  # 8 weeks forecast
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
            'dates': df['ds'].astype(str).tolist() + forecast['ds'].tail(8).astype(str).tolist(),
            'actual': df['y'].tolist() + [None] * 8,
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