import pandas as pd
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import keras_tuner as kt
import matplotlib.pyplot as plt

# Preprocess and prepare the dataset
def preprocess_and_prepare_dataset():
    # Load the dataset
    url = "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv"
    data = pd.read_csv(url)
    data['Date_reported'] = pd.to_datetime(data['Date_reported'])

    # Aggregate data globally by date
    global_data = data.groupby('Date_reported').agg({'New_cases': 'sum'}).reset_index()

    # Apply log transformation to stabilize variance
    global_data['New_cases'] = np.log1p(global_data['New_cases'])

    # Normalize the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    global_data['New_cases'] = scaler.fit_transform(global_data['New_cases'].values.reshape(-1, 1))

    return global_data, scaler

# Create consistent features for all models
def create_features(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
    X_flat = X.reshape(X.shape[0], -1)  # Flatten for non-sequential models
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM/GRU
    return X, X_flat, y

# Define the LSTM/GRU model building function for Keras Tuner
def build_lstm_gru_model(hp):
    model = Sequential()

    # Tuning sequence length
    sequence_length = hp.Int('sequence_length', min_value=30, max_value=120, step=10)

    # Input layer
    model.add(Input(shape=(sequence_length, 1)))

    for i in range(hp.Int('num_layers', 1, 3)):
        layer_type = hp.Choice(f'layer_type_{i}', ['LSTM', 'GRU'])
        units = hp.Int(f'units_{i}', min_value=10, max_value=100, step=10)

        if layer_type == 'LSTM':
            model.add(LSTM(units=units, return_sequences=(i < hp.Int('num_layers', 1, 3) - 1)))
        else:
            model.add(GRU(units=units, return_sequences=(i < hp.Int('num_layers', 1, 3) - 1)))

        # Optional normalization layer
        if hp.Boolean(f'normalization_{i}'):
            model.add(BatchNormalization())

        # Dropout layer
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'dense_units_{i}', min_value=10, max_value=100, step=10)))

    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
                      learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='mean_absolute_percentage_error')

    return model

# Tuning the LSTM/GRU model
lstm_gru_tuner = kt.BayesianOptimization(
    build_lstm_gru_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=2,
    directory='lstm_gru_dir',
    project_name='lstm_gru_tuning'
)

# Tuning ARIMA model parameters
def tune_arima_params(train, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    return model_fit

# Function to train and predict using RandomForestRegressor
def random_forest_model(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function to train and predict using XGBRegressor
def xgboost_model(X_train, y_train, X_test):
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Define the goal MAPE
goal_mape = 10.0  # Adjust as needed

# Preprocess the dataset
global_data, scaler = preprocess_and_prepare_dataset()

# Initialize the sequence length for the first run
initial_sequence_length = 60  # You can adjust this value
X, X_flat, y = create_features(global_data['New_cases'].values, initial_sequence_length)

# Initialize the ARIMA order
arima_order = (5, 1, 0)  # Adjust these values as necessary for your data

# Simulation of the GAN-style competition
for iteration in range(10):
    print(f"Iteration {iteration + 1}")

    # Tune LSTM/GRU model
    lstm_gru_tuner.search(X, y, epochs=10, validation_split=0.2, batch_size=32)
    best_lstm_gru_model = lstm_gru_tuner.get_best_models(num_models=1)[0]

    # Retrieve the best sequence length from the hyperparameters
    best_sequence_length = lstm_gru_tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values['sequence_length']

    # Recreate sequences using the best sequence length
    X, X_flat, y = create_features(global_data['New_cases'].values, best_sequence_length)

    # ARIMA training and tuning
    train_data = global_data['New_cases'].values
    arima_model = tune_arima_params(train_data, arima_order)

    # Random Forest training and prediction
    rf_predictions = random_forest_model(X_flat, y, X_flat)

    # XGBoost training and prediction
    xgb_predictions = xgboost_model(X_flat, y, X_flat)

    # LSTM/GRU prediction
    lstm_gru_predictions = best_lstm_gru_model.predict(X)

    # ARIMA prediction
    arima_predictions = arima_model.predict(start=len(train_data), end=len(train_data)+len(y)-1)

    # Calculate performance metrics (e.g., MAPE)
    lstm_gru_mape = np.mean(np.abs((y.flatten() - lstm_gru_predictions.flatten()) / y.flatten())) * 100
    arima_mape = np.mean(np.abs((y.flatten() - arima_predictions.flatten()) / y.flatten())) * 100
    rf_mape = np.mean(np.abs((y.flatten() - rf_predictions.flatten()) / y.flatten())) * 100
    xgb_mape = np.mean(np.abs((y.flatten() - xgb_predictions.flatten()) / y.flatten())) * 100

    print(f"LSTM/GRU MAPE: {lstm_gru_mape}")
    print(f"ARIMA MAPE: {arima_mape}")
    print(f"Random Forest MAPE: {rf_mape}")
    print(f"XGBoost MAPE: {xgb_mape}")

    # Check if any model meets the goal MAPE
    if lstm_gru_mape <= goal_mape or arima_mape <= goal_mape or rf_mape <= goal_mape or xgb_mape <= goal_mape:
        print(f"Goal MAPE of {goal_mape}% reached in iteration {iteration + 1}")
        break

    # Update ARIMA order based on performance (simple strategy, refine as needed)
    if lstm_gru_mape < arima_mape:
        arima_order = (arima_order[0] + 1, arima_order[1], arima_order[2])
    else:
        arima_order = (arima_order[0], arima_order[1], arima_order[2] + 1)

    # Optional: Break the loop early if the performance stabilizes
    if abs(lstm_gru_mape - arima_mape) < 0.1:
        print("Performance has stabilized, stopping early.")
        break

# Final evaluation with the best models
best_lstm_gru_model.fit(X, y, epochs=100, batch_size=32)
final_lstm_gru_predictions = best_lstm_gru_model.predict(X[-7:])
final_arima_predictions = arima_model.predict(start=len(global_data['New_cases'])-7, end=len(global_data['New_cases'])-1)
final_rf_predictions = random_forest_model(X_flat[:-7], y[:-7], X_flat[-7:])
final_xgb_predictions = xgboost_model(X_flat[:-7], y[:-7], X_flat[-7:])

# Inverse transform predictions and compare
final_lstm_gru_predictions = scaler.inverse_transform(np.array(final_lstm_gru_predictions).reshape(-1, 1)).flatten()
final_arima_predictions = scaler.inverse_transform(np.array(final_arima_predictions).reshape(-1, 1)).flatten()
final_rf_predictions = scaler.inverse_transform(np.array(final_rf_predictions).reshape(-1, 1)).flatten()
final_xgb_predictions = scaler.inverse_transform(np.array(final_xgb_predictions).reshape(-1, 1)).flatten()

# Apply exponential function to reverse the log transformation
final_lstm_gru_predictions = np.expm1(final_lstm_gru_predictions)
final_arima_predictions = np.expm1(final_arima_predictions)
final_rf_predictions = np.expm1(final_rf_predictions)
final_xgb_predictions = np.expm1(final_xgb_predictions)

# Get the actual cases for the last 7 days
actual_cases = np.expm1(scaler.inverse_transform(global_data['New_cases'].values[-7:].reshape(-1, 1)).flatten())

# Calculate MAPE for the final predictions
final_lstm_gru_mape = np.mean(np.abs((actual_cases - final_lstm_gru_predictions) / actual_cases)) * 100
final_arima_mape = np.mean(np.abs((actual_cases - final_arima_predictions) / actual_cases)) * 100
final_rf_mape = np.mean(np.abs((actual_cases - final_rf_predictions) / actual_cases)) * 100
final_xgb_mape = np.mean(np.abs((actual_cases - final_xgb_predictions) / actual_cases)) * 100

# Print the final MAPE values
print("\nFinal MAPE values:")
print(f"LSTM/GRU Final MAPE: {final_lstm_gru_mape}")
print(f"ARIMA Final MAPE: {final_arima_mape}")
print(f"Random Forest Final MAPE: {final_rf_mape}")
print(f"XGBoost Final MAPE: {final_xgb_mape}")

# Create a DataFrame to compare actual vs predicted
comparison_df = pd.DataFrame({
    'Date': global_data['Date_reported'].values[-7:],
    'Actual': actual_cases,
    'LSTM_GRU_Predicted': final_lstm_gru_predictions,
    'ARIMA_Predicted': final_arima_predictions,
    'Random_Forest_Predicted': final_rf_predictions,
    'XGBoost_Predicted': final_xgb_predictions
})

# Display the comparison DataFrame
print(comparison_df)