import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def fetch_and_clean_data():
    print("Fetching and cleaning data...")
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(url)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
    df['Date_reported'] = pd.to_datetime(df['Date_reported'])
    df = df.sort_values('Date_reported')
    
    df_global = df.groupby('Date_reported')['New_cases'].sum().reset_index()
    df_global['New_cases'] = df_global['New_cases'].fillna(0).clip(lower=0)
    
    # Add time-based features
    df_global['dayofweek'] = df_global['Date_reported'].dt.dayofweek
    df_global['month'] = df_global['Date_reported'].dt.month
    df_global['day'] = df_global['Date_reported'].dt.day
    
    # Calculate rolling averages
    df_global['7day_avg'] = df_global['New_cases'].rolling(window=7).mean()
    df_global['30day_avg'] = df_global['New_cases'].rolling(window=30).mean()
    
    # Fill NaN values in rolling averages with the mean
    df_global['7day_avg'] = df_global['7day_avg'].fillna(df_global['7day_avg'].mean())
    df_global['30day_avg'] = df_global['30day_avg'].fillna(df_global['30day_avg'].mean())
    
    print(f"Data cleaned. Shape: {df_global.shape}")
    return df_global

def prepare_data(df):
    features = ['dayofweek', 'month', 'day', '7day_avg', '30day_avg']
    X = df[features]
    y = df['New_cases']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        predictions = np.maximum(model.predict(X_test), 0)  # Ensure non-negative predictions
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

def make_future_predictions(df, models, days=7):
    last_date = df['Date_reported'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    future_df = pd.DataFrame({'Date_reported': future_dates})
    future_df['dayofweek'] = future_df['Date_reported'].dt.dayofweek
    future_df['month'] = future_df['Date_reported'].dt.month
    future_df['day'] = future_df['Date_reported'].dt.day
    
    # Use the average of the last 7 days for future predictions
    future_df['7day_avg'] = df['New_cases'].tail(7).mean()
    future_df['30day_avg'] = df['New_cases'].tail(30).mean()
    
    scaler = StandardScaler()
    scaler.fit(df[['dayofweek', 'month', 'day', '7day_avg', '30day_avg']])
    future_scaled = scaler.transform(future_df[['dayofweek', 'month', 'day', '7day_avg', '30day_avg']])
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = np.maximum(model.predict(future_scaled), 0)  # Ensure non-negative predictions
    
    future_df['RF_Predicted'] = predictions['Random Forest']
    future_df['XGB_Predicted'] = predictions['XGBoost']
    
    return future_df

def main():
    df = fetch_and_clean_data()
    if df is None:
        return
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    rf_model, xgb_model = train_models(X_train, y_train)
    models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    
    print("\nModel Evaluation:")
    evaluate_models(models, X_test, y_test)
    
    future_predictions = make_future_predictions(df, models)
    
    print("\nFuture Predictions:")
    print(future_predictions.to_string(index=False))

    # Save predictions to JSON
    predictions_dict = {
        'dates': future_predictions['Date_reported'].dt.strftime('%Y-%m-%d').tolist(),
        'rf_predictions': future_predictions['RF_Predicted'].tolist(),
        'xgb_predictions': future_predictions['XGB_Predicted'].tolist()
    }
    
    import json
    with open('covid_predictions.json', 'w') as f:
        json.dump(predictions_dict, f)

if __name__ == "__main__":
    main()