import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sma_signals(strategy_data, ticker=""):
    r"""
    Plot close price, SMA strategy net value, and signal indicator for visualization.

    Parameters
    ----------
    strategy_data : pd.DataFrame
        Strategy result data containing 'Close', 'SMA_Value', and 'Indicator'.
    ticker : str, optional
        Stock ticker name to use in the title. Default is empty string.

    Returns
    -------
    None
        Displays the figure directly.
    """
    # Create subplots with 2 rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                   gridspec_kw={'height_ratios': [2.5, 1]},
                                   sharex=True)
    fig.subplots_adjust(right=0.85)  # Leave space on the right for legend

    # Plot closing price and SMA strategy net value
    ax1.plot(strategy_data.index, strategy_data['Close'], label='Close Price', color='navy')
    ax1.plot(strategy_data.index, strategy_data['SMA_Value'], label='SMA Value', color='pink', alpha=0.6)
    ax1.set_ylabel('Close Price')
    ax1.set_title(f"{ticker} - SMA Strategy")
    ax1.grid(True, alpha=0.5, linestyle='--')
    ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    # Plot signal indicator region
    indicator = strategy_data['Indicator'].copy()
    ax2.plot(strategy_data.index, indicator, label='SMA Indicator', color='blue', linewidth=1.0)
    ax2.fill_between(strategy_data.index, 0, indicator.where(indicator > 0),
                     facecolor='green', alpha=0.6, label='Positive Indicator')
    ax2.fill_between(strategy_data.index, 0, indicator.where(indicator < 0),
                     facecolor='red', alpha=0.6, label='Negative Indicator')
    ax2.set_ylabel('SMA Indicator')
    ax2.grid(True, alpha=0.5, linestyle='--')
    ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    # Format x-axis as monthly date
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x')
    ax1.tick_params(labelbottom=True)

    plt.show()


def plot_rsi_signals(strategy_data, ticker=""):
    r"""
    Plot close price, RSI strategy value, and RSI signals with colored regions.

    Parameters
    ----------
    strategy_data : pd.DataFrame
        Strategy result data containing 'Close', 'RSI_Value', and 'RSI_Signal'.
    ticker : str, optional
        Stock ticker name to use in the title. Default is empty string.

    Returns
    -------
    None
        Displays the figure directly.
    """
    # Create RSI-specific subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                   gridspec_kw={'height_ratios': [2.5, 1]},
                                   sharex=True)
    fig.subplots_adjust(right=0.85)

    # Plot close price and RSI strategy net value
    ax1.plot(strategy_data.index, strategy_data['Close'], label='Close Price', color='navy')
    ax1.plot(strategy_data.index, strategy_data['RSI_Value'], label='RSI Value', color='pink', alpha=0.6)
    ax1.set_ylabel('Close Price')
    ax1.set_title(f"{ticker} - RSI Strategy")
    ax1.grid(True, alpha=0.5, linestyle='--')
    ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    # Plot signal fill region (buy/sell)
    signal = strategy_data['RSI_Signal'].copy()
    ax2.plot(strategy_data.index, signal, label='RSI Signal', color='blue', linewidth=1.0)
    ax2.fill_between(strategy_data.index, 0, signal.where(signal > 0),
                     facecolor='green', alpha=0.6, label='Buy Signal')
    ax2.fill_between(strategy_data.index, 0, signal.where(signal < 0),
                     facecolor='red', alpha=0.6, label='Sell Signal')
    ax2.set_ylabel('RSI Signal')
    ax2.grid(True, alpha=0.5, linestyle='--')
    ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as year-month
    ax2.tick_params(axis='x')
    ax1.tick_params(labelbottom=True)

    plt.show()


def plot_strategy_signals(strategy_data, value_col, signal_label="Strategy", ticker=""):
    r"""
    General-purpose strategy plot with close price, strategy net value, and signal region.

    Parameters
    ----------
    strategy_data : pd.DataFrame
        DataFrame containing 'Close', value column, and 'Signal'.
    value_col : str
        Column name of the strategy value to be plotted.
    signal_label : str, optional
        Label prefix for signal and value line. Default is "Strategy".
    ticker : str, optional
        Stock ticker name to include in the title.

    Returns
    -------
    None
        Displays the plot.
    """
    # Set up general strategy plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                   gridspec_kw={'height_ratios': [2.5, 1]},
                                   sharex=True)
    fig.subplots_adjust(right=0.85)

    # Price and net value plot
    ax1.plot(strategy_data.index, strategy_data['Close'], label='Close Price', color='navy')
    ax1.plot(strategy_data.index, strategy_data[value_col], label=f'{signal_label} Value', color='pink', alpha=0.6)
    ax1.set_ylabel('Close Price')
    ax1.set_title(f"{ticker} - {signal_label} Strategy")
    ax1.grid(True, alpha=0.5, linestyle='--')
    ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    # Signal indicator region
    signal = strategy_data['Signal']
    ax2.plot(strategy_data.index, signal, label=f'{signal_label} Signal', color='blue', linewidth=1.0)
    ax2.fill_between(strategy_data.index, 0, signal.where(signal > 0),
                     facecolor='green', alpha=0.6, label='Buy Region')
    ax2.fill_between(strategy_data.index, 0, signal.where(signal < 0),
                     facecolor='red', alpha=0.6, label='Sell Region')
    ax2.set_ylabel(f'{signal_label} Signal')
    ax2.grid(True, alpha=0.5, linestyle='--')
    ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.show()


def plot_sharpe_heatmap(score_func,
                        x_range=range(3, 31),
                        y_range=range(10, 61),
                        best_params=None,
                        title="Sharpe Ratio Heatmap",
                        xlabel="X Parameter",
                        ylabel="Y Parameter",
                        enforce_order=False):
    r"""
    Generate a Sharpe ratio heatmap for a grid of two strategy parameters.

    Parameters
    ----------
    score_func : function
        A function that accepts two integers and returns a Sharpe ratio.
    x_range : range
        Values for horizontal axis (e.g. short window).
    y_range : range
        Values for vertical axis (e.g. long window).
    best_params : tuple, optional
        Coordinates of the best parameter pair for highlighting.
    title : str
        Title of the heatmap.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    enforce_order : bool
        If True, skips cases where x >= y. Useful for ordered pairs like short < long.

    Returns
    -------
    None
        Displays the heatmap.
    """
    results = []  # Store evaluation results
    for x in x_range:
        for y in y_range:
            if enforce_order and x >= y:
                sharpe = np.nan  # Skip invalid combinations
            else:
                try:
                    sharpe = score_func(x, y)  # Evaluate Sharpe score for params
                except Exception as e:
                    print(f"Error at ({x},{y}): {e}")
                    sharpe = np.nan
            results.append((x, y, sharpe))  # Store result

    # Create pivot table for heatmap input
    df = pd.DataFrame(results, columns=['x', 'y', 'sharpe'])
    pivot = df.pivot_table(index='y', columns='x', values='sharpe', aggfunc='mean')
    pivot.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values

    # Generate heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot, annot=False, cmap='viridis', linewidths=0.3, linecolor='gray')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Mark best point with red star if applicable
    if best_params is not None:
        x_best, y_best = int(best_params[0]), int(best_params[1])
        x_labels = list(pivot.columns)
        y_labels = list(pivot.index)
        if x_best in x_labels and y_best in y_labels:
            x_idx = x_labels.index(x_best)
            y_idx = y_labels.index(y_best)
            plt.plot(x_idx + 0.5, y_idx + 0.5, 'r*', markersize=12)
            plt.text(x_idx + 0.5, y_idx + 0.3, f"({x_best},{y_best})", color='red')
        else:
            print(f"Warning: Best point ({x_best},{y_best}) not in heatmap range.")

    # Print debug info
    print("Heatmap grid shape:", pivot.shape)
    print("X-axis labels:", list(pivot.columns))
    print("Y-axis labels:", list(pivot.index))

    plt.tight_layout()
    plt.show()


def plot_metric_percentile(metric_series, best_value, title, xlabel):
    r"""
    Plot a histogram of a performance metric with the best value highlighted.

    Parameters
    ----------
    metric_series : pd.Series
        Series of metric values (e.g., Sharpe ratios).
    best_value : float
        The best score obtained, marked on the histogram.
    title : str
        Title of the chart.
    xlabel : str
        Label for the x-axis.

    Returns
    -------
    None
        Displays the plot.
    """
    sns.histplot(metric_series, kde=True)  # Histogram with KDE
    plt.axvline(best_value, color='red', linestyle='--', label=f'Best = {best_value:.4f}')  # Mark best value
    percentile = (metric_series < best_value).mean() * 100  # Compute percentile rank
    plt.title(f"{title}\nBest parameter is at {percentile:.2f} percentile")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
