import os
import logging
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, period: str = "90d", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch stock data from Yahoo Finance for a given ticker."""
    logger.info(f"Fetching data for {ticker}...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            logger.error(f"No data returned for {ticker}.")
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def process_stock_data(df: pd.DataFrame, rsi_period: int = 14) -> Optional[pd.DataFrame]:
    """Clean and process stock data, adding RSI indicator."""
    if df is None or df.empty:
        logger.error("No data to process.")
        return None

    # Normalize or flatten column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    logger.debug(f"Available columns: {df.columns.tolist()}")

    # Try to find the correct 'close' column
    close_col = next((col for col in df.columns if 'close' in col), None)
    if not close_col:
        logger.error(f"Missing required 'close' column. Available columns: {df.columns.tolist()}")
        return None

    # Drop rows with NaNs in close
    if df[close_col].isna().any():
        logger.warning(f"Missing values found in '{close_col}' column. Dropping rows with NaN values.")
        df = df.dropna(subset=[close_col])

    if len(df) < rsi_period:
        logger.error(f"Insufficient data points ({len(df)}) for RSI calculation with period {rsi_period}.")
        return None

    try:
        rsi = RSIIndicator(df[close_col], window=rsi_period)
        df['rsi'] = rsi.rsi()
        df = df.dropna(subset=['rsi'])
        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return None

def save_data(df: pd.DataFrame, ticker: str, data_dir: str = 'data') -> None:
    """Save processed stock data to a CSV file."""
    if df is None or df.empty:
        logger.error(f"No data to save for {ticker}.")
        return

    try:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{ticker}_data.csv")
        if os.path.exists(file_path):
            logger.warning(f"File {file_path} already exists and will be overwritten.")
        df.to_csv(file_path, index=True)
        logger.info(f"Data for {ticker} saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")

def run(tickers: list[str], data_dir: str = 'data') -> None:
    """Run the full pipeline: fetch, process, and save stock data for a list of tickers."""
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if df is None:
            logger.warning(f"Skipping {ticker} due to fetch failure.")
            continue

        processed_df = process_stock_data(df)
        if processed_df is None:
            logger.warning(f"Skipping {ticker} due to processing failure.")
            continue

        save_data(processed_df, ticker, data_dir)

if __name__ == "__main__":
    tickers = ['RELIANCE.NS', 'TCS.NS']
    run(tickers)
