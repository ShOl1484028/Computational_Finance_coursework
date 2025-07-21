import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker, start, end, interval='1d', filename=None, reload=False):
    r"""
    Fetch historical price data for a single stock, with optional local caching
    and automatic date index formatting.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    start : str
        Start date in 'YYYY-MM-DD' format (e.g., '2020-01-01').
    end : str
        End date in 'YYYY-MM-DD' format (e.g., '2024-01-01').
    interval : str, optional
        Data frequency, default is '1d'. Other options include '1h', '1wk', etc.
    filename : str, optional
        File path to save or load data. If None, defaults to 'data/{ticker}.xlsx'.
    reload : bool, optional
        If True, forces redownload of data even if file exists. Default is False.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing historical stock data with datetime index.

    Notes
    -----
    The function checks if a local file exists. If so and `reload=False`, it reads
    from the file; otherwise, it downloads the data using Yahoo Finance API.
    """
    if filename is None:
        filename = f'data/{ticker}.xlsx'  # Default path for storing data

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create folder if not exists

    # Load from cache if file exists and reload not requested
    if os.path.exists(filename) and not reload:
        print(f"Loading from local file: {filename}")
        df = pd.read_excel(filename)
    else:
        print(f"Downloading data for: {ticker}")
        df = yf.download(ticker, start=start, end=end, interval=interval)

        # Flatten multi-level column names if present (e.g., OHLCV)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.to_excel(filename)
        print(f"Data saved to: {filename}")

    # ----- Ensure datetime index is set correctly -----
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])  # Convert column to datetime
        df.set_index('Date', inplace=True)
    elif df.index.name != 'Date':
        df.index = pd.to_datetime(df.index)  # Ensure index is datetime
        df.index.name = 'Date'

    return df


if __name__ == '__main__':
    df = fetch_stock_data('UNH', '2013-04-01', '2025-04-02')  # Example usage
