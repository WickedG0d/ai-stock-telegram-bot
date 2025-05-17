import os
from data_collector import *
from preprocessing import *
from lstm_model import *

def main():
    """Main function to run the entire pipeline."""
    tickers = ['RELIANCE.NS', 'TCS.NS']  # List of stock tickers
    data_dir = 'data'  # Directory to save data
    model_save_path = 'models'  # Directory to save the model

    # Step 1: Collect and save stock data
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if df is not None:
            processed_df = process_stock_data(df)
            if processed_df is not None:
                save_data(processed_df, ticker, data_dir)
    
    # Step 2: Train the LSTM model
    for ticker in tickers:
        ticker_data_path = os.path.join(data_dir, f"{ticker}_data.csv")
        if os.path.exists(ticker_data_path):
            df = pd.read_csv(ticker_data_path, index_col="Date", parse_dates=True)
            train_lstm_model(df, model_save_path)  # Pass model_save_path as the file_path argument

if __name__ == "__main__":
    main()
