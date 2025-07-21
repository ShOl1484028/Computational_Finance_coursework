import numpy as np
import pandas as pd
from itertools import product


def sma_strategy(data, short_window=5, long_window=10):
    r"""
    Apply Simple Moving Average crossover strategy to price data.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with a 'Close' column.
    short_window : int
        Short moving average window.
    long_window : int
        Long moving average window.

    Returns
    -------
    pd.DataFrame
        DataFrame with strategy signals and performance metrics.
    """
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")

    df = data.copy()
    initial_close = df['Close'].iloc[0]  # used for rebasing value

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))  # daily log returns
    df['SMA_Short'] = df['Close'].rolling(short_window).mean()  # short-term SMA
    df['SMA_Long'] = df['Close'].rolling(long_window).mean()    # long-term SMA
    df['Indicator'] = df['SMA_Short'] - df['SMA_Long']  # signal line

    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)  # position: long or short
    df['SMA_Return'] = df['Signal'].shift(1) * df['Log_Return']  # strategy return
    df['Cumulative_SMA'] = df['SMA_Return'].cumsum()
    df['SMA_Value'] = initial_close * np.exp(df['Cumulative_SMA'])  # rebased equity curve

    df['Strategy_Return'] = df['SMA_Return']
    df['Strategy_Value'] = df['SMA_Value']
    df.dropna(inplace=True)
    return df


def rsi_strategy(data, rsi_period=14, threshold=30):
    r"""
    Apply RSI-based strategy to generate buy/sell signals.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with a 'Close' column.
    rsi_period : int
        Lookback window for RSI calculation.
    threshold : float
        Entry/exit threshold for RSI.

    Returns
    -------
    pd.DataFrame
        DataFrame with strategy signals and performance metrics.
    """
    df = data.copy()
    initial_close = df['Close'].iloc[0]  # used for rebasing

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))  # daily log return

    delta = df['Close'].diff()  # daily price change
    gain = np.where(delta > 0, delta, 0)  # positive gains
    loss = np.where(delta < 0, -delta, 0)  # absolute losses
    gain_ema = pd.Series(gain, index=df.index).ewm(span=rsi_period, adjust=False).mean()  # EMA of gains
    loss_ema = pd.Series(loss, index=df.index).ewm(span=rsi_period, adjust=False).mean()  # EMA of losses

    rs = gain_ema / loss_ema  # relative strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    df['RSI'] = rsi

    df['Signal'] = 0  # initialize signal
    df.loc[rsi < threshold, 'Signal'] = 1  # long signal
    df.loc[rsi > 100 - threshold, 'Signal'] = -1  # short signal

    df['RSI_Return'] = df['Signal'].shift(1) * df['Log_Return']
    df['RSI_Value'] = initial_close * np.exp(df['RSI_Return'].cumsum())
    df['Strategy_Return'] = df['RSI_Return']
    df['Strategy_Value'] = df['RSI_Value']
    df['Indicator'] = df['RSI']  # for plotting
    df.dropna(inplace=True)
    return df


def reversion_strategy(data, short_window, long_window, entry_dev=-0.01):
    r"""
    Mean-reversion strategy based on deviation between short and long SMA.

    Parameters
    ----------
    data : pd.DataFrame
        Input price data.
    short_window : int
        Short SMA window.
    long_window : int
        Long SMA window.
    entry_dev : float
        Deviation threshold for entry.

    Returns
    -------
    pd.DataFrame
        DataFrame with strategy signals and metrics.
    """
    df = data.copy()
    initial_close = df['Close'].iloc[0]  # baseline value

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))  # log return
    df['SMA_short'] = df['Close'].rolling(short_window).mean()
    df['SMA_long'] = df['Close'].rolling(long_window).mean()
    df['Dev'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']  # relative deviation

    df['Signal'] = 0
    position = 0  # 1 = long, 0 = flat

    for i in range(1, len(df)):
        if position == 0 and df['Dev'].iloc[i - 1] < entry_dev:
            position = 1  # enter long
        elif position == 1 and df['Dev'].iloc[i - 1] >= 0:
            position = 0  # exit
        df.at[df.index[i], 'Signal'] = position

    df['REV_Return'] = df['Signal'].shift(1) * df['Log_Return']
    df['REV_Value'] = initial_close * np.exp(df['REV_Return'].cumsum())
    df['Strategy_Return'] = df['REV_Return']
    df['Strategy_Value'] = df['REV_Value']
    df['Indicator'] = df['Dev']  # used for plotting
    df.dropna(inplace=True)
    return df


def bollinger_strategy(data, window=20, num_std=2.0, entry_factor=0.3, exit_buffer=0):
    r"""
    Bollinger Band strategy with configurable entry/exit sensitivity.

    Parameters
    ----------
    data : pd.DataFrame
        Price data.
    window : int
        Moving average window.
    num_std : float
        Number of standard deviations for bands.
    entry_factor : float
        Fraction of band width to trigger entry.
    exit_buffer : float
        Fraction of band width to trigger exit.

    Returns
    -------
    pd.DataFrame
        DataFrame with signals and performance metrics.
    """
    df = data.copy()
    initial_close = df['Close'].iloc[0]

    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))  # daily log return
    df['MA'] = df['Close'].rolling(window).mean()  # rolling mean
    df['STD'] = df['Close'].rolling(window).std()  # rolling std

    # Define dynamic entry/exit bands
    entry_upper = df['MA'] + entry_factor * num_std * df['STD']
    entry_lower = df['MA'] - entry_factor * num_std * df['STD']
    exit_upper = df['MA'] + exit_buffer * df['STD']
    exit_lower = df['MA'] - exit_buffer * df['STD']

    # Signal generation based on bands
    signal = np.where(df['Close'] > entry_upper, -1,
                      np.where(df['Close'] < entry_lower, 1, np.nan))

    signal = pd.Series(signal, index=df.index).ffill()  # forward fill open positions
    signal[(df['Close'] < exit_upper) & (df['Close'] > exit_lower)] = 0  # flatten in MA zone

    df['Signal'] = signal.fillna(0)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Log_Return']  # apply signal
    df['Strategy_Value'] = initial_close * np.exp(df['Strategy_Return'].cumsum())
    df['Indicator'] = df['Close']  # can plot with price overlay
    df.dropna(inplace=True)
    return df


def evaluate_param_performance(test_data, strategy_func, param_grid):
    r"""
    Evaluate parameter combinations using a grid for a given strategy.

    Parameters
    ----------
    test_data : pd.DataFrame
        Data for evaluation.
    strategy_func : function
        Strategy function accepting **kwargs.
    param_grid : dict
        Dictionary of parameters and ranges.

    Returns
    -------
    pd.DataFrame
        DataFrame with Sharpe ratios and returns for each parameter combo.
    """
    sharpe_list = []
    return_list = []
    param_list = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        param_dict = dict(zip(keys, combo))  # build parameter set

        try:
            df = strategy_func(test_data, **param_dict)  # run strategy
            rets = df['Strategy_Return']
            sharpe = rets.mean() / rets.std() * np.sqrt(252)  # annualized Sharpe
            annual_ret = rets.mean() * 252

            if np.isfinite(sharpe):  # filter out NaN or inf results
                sharpe_list.append(sharpe)
                return_list.append(annual_ret)
                param_list.append(param_dict)
        except Exception as e:
            print(f"Passing param {param_dict}: {e}")
            continue

    # Combine results into final output table
    return pd.DataFrame(
        [dict(**params, sharpe=s, annual_return=r)
         for params, s, r in zip(param_list, sharpe_list, return_list)]
    )
