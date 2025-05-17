import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load the preprocessed stock data."""
    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

def prepare_data(df: pd.DataFrame, stock_name: str, feature_columns: list = None, target_column: str = 'close', sequence_length: int = 60):
    """Prepare the data for LSTM by scaling and creating sequences."""
    if feature_columns is None:
        feature_columns = [f'close_{stock_name}']

    if target_column == 'close':
        target_column = f'close_{stock_name}'

    # Check if required columns exist
    if not all(col in df.columns for col in feature_columns):
        logger.error(f"Missing required columns: {feature_columns}")
        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns].values)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])  # The feature (close) over the last 60 days
        y.append(scaled_data[i, 0])  # The target is the closing price on the next day

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input
    
    return X, y, scaler

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer for the predicted close price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(df: pd.DataFrame, stock_name: str, file_path: str, sequence_length: int = 60, test_size: float = 0.2):
    """Train the LSTM model on the stock data."""
    # Prepare data
    X, y, scaler = prepare_data(df, stock_name, sequence_length=sequence_length)
    if X is None or y is None or scaler is None:
        logger.error("Data preparation failed, cannot proceed with training.")
        return
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Build LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the model and scaler for future predictions
    model.save(file_path + '/lstm_stock_model.h5')
    joblib.dump(scaler, file_path + '/scaler.pkl')
    logger.info(f"Model saved to {file_path}/lstm_stock_model.h5")
    logger.info(f"Scaler saved to {file_path}/scaler.pkl")

def predict_stock_price(model_path: str, scaler_path: str, data: pd.DataFrame, stock_name: str, sequence_length: int = 60):
    """Predict the stock price for the next day using the trained LSTM model."""
    # Load model and scaler
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare the last 60 days of data for prediction
    recent_data = data.tail(sequence_length)[[f'close_{stock_name}']]
    scaled_data = scaler.transform(recent_data.values)
    X_predict = scaled_data.reshape(1, sequence_length, 1)  # Reshape for LSTM input
    
    # Predict the next day's closing price
    predicted_price = model.predict(X_predict)
    predicted_price = scaler.inverse_transform(predicted_price)  # Inverse transform to original scale
    
    return predicted_price[0][0]

if __name__ == "__main__":
    # Example usage
    data_file_path = 'data/RELIANCE.NS_data.csv'  # Change to the path of your processed data
    model_save_path = 'models'  # Directory to save the model and scaler
    os.makedirs(model_save_path, exist_ok=True)
    
    stock_name = 'reliance.ns'  # Add the stock name here
    
    # Load data
    df = load_data(data_file_path)
    if df is not None:
        # Train and save the model
        train_lstm_model(df, stock_name, model_save_path)
        
        # Example prediction for the next day's stock price
        predicted_price = predict_stock_price(
            model_path=f"{model_save_path}/lstm_stock_model.h5",
            scaler_path=f"{model_save_path}/scaler.pkl",
            data=df,
            stock_name=stock_name
        )
        logger.info(f"Predicted stock price for the next day: {predicted_price}")
