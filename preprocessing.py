import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load stock data from a CSV file."""
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df: pd.DataFrame, feature_col: str = 'close', rsi_col: str = 'rsi', window_size: int = 60):
    """Preprocess data for LSTM model by scaling, creating sequences, and splitting into train/test sets."""
    if df is None or df.empty:
        print("No data to preprocess.")
        return None, None, None

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_col, rsi_col]])  # Scaling close and RSI

    # Create data sequences for LSTM (using window_size)
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(scaled_data[i - window_size:i])  # Features (last `window_size` days)
        y.append(scaled_data[i, 0])  # Target is the 'close' price on the next day

    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler

