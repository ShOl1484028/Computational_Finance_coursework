import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_single_path(t_axis, price_series):
    r"""
    Plot a single simulated price path over time.

    Parameters
    ----------
    t_axis : array-like
        Time axis array (e.g., trading days).
    price_series : array-like
        Price data corresponding to t_axis.

    Returns
    -------
    None
        Displays the figure.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(t_axis, price_series)  # Line plot of the path
    plt.xlabel("Trading Day")
    plt.ylabel("Price")
    plt.title("Simulated Daily Spot Price of Acme Co (1 Year)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multiple_paths(t_axis, price_matrix, title="Simulated Spot Price Paths",
                        max_lines=None, ylabel="Price"):
    r"""
    Plot multiple simulation paths (e.g., spot, delta, portfolio value).

    Parameters
    ----------
    t_axis : array-like
        Time axis of shape [T+1].
    price_matrix : np.ndarray
        Matrix of shape [n_paths, T+1] containing path data.
    title : str, optional
        Title of the plot.
    max_lines : int, optional
        Maximum number of paths to show. If None, show all.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    None
        Displays the figure.
    """
    plt.figure(figsize=(10, 5))
    n_paths = price_matrix.shape[0]
    n_show = n_paths if max_lines is None else min(n_paths, max_lines)
    for i in range(n_show):
        plt.plot(t_axis, price_matrix[i], alpha=1, linewidth=1)  # Plot each path
    plt.xlabel("Trading Day")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(data, xlabel, title):
    r"""
    Plot a histogram with KDE for the given data.

    Parameters
    ----------
    data : array-like
        Data to be plotted.
    xlabel : str
        Label for the x-axis.
    title : str
        Plot title.

    Returns
    -------
    None
        Displays the histogram.
    """
    sns.histplot(data, kde=True)  # Histogram + KDE
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_delta_paths(t_axis, delta_matrix, max_lines=50):
    r"""
    Plot multiple delta paths over time.

    Parameters
    ----------
    t_axis : array-like
        Time axis array.
    delta_matrix : np.ndarray
        Matrix of shape [n_paths, T+1] with delta values.
    max_lines : int, optional
        Maximum number of lines to plot. Default is 50.

    Returns
    -------
    None
        Displays the figure.
    """
    plt.figure(figsize=(10, 5))
    for i in range(min(delta_matrix.shape[0], max_lines)):
        plt.plot(t_axis, delta_matrix[i], alpha=0.5, linewidth=0.5)  # Plot each delta path
    plt.xlabel("Year (Trading Time)")
    plt.ylabel("Portfolio Delta")
    plt.title("Simulated Daily Delta of Portfolio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_distribution(
    data,
    xlabel="Value",
    title="Distribution",
    show_mean=True,
    show_percentiles=(1, 5, 99),
    figsize=(8, 4),
    color=None,
    kde=True,
    bins=50,
    grid=True
):
    r"""
    Plot a distribution with optional mean and percentile annotations.

    Parameters
    ----------
    data : np.ndarray
        1D array of values to plot.
    xlabel : str
        Label for the x-axis.
    title : str
        Plot title.
    show_mean : bool, optional
        Whether to display a vertical mean line. Default is True.
    show_percentiles : tuple of int, optional
        Percentile values to mark. Default is (1, 5, 99).
    figsize : tuple, optional
        Figure size. Default is (8, 4).
    color : str, optional
        Color for the histogram bars. Default is None.
    kde : bool, optional
        Whether to overlay a KDE curve. Default is True.
    bins : int, optional
        Number of bins for the histogram. Default is 50.
    grid : bool, optional
        Whether to display grid lines. Default is True.

    Returns
    -------
    None
        Displays the plot.
    """
    plt.figure(figsize=figsize)
    sns.histplot(data, kde=kde, bins=bins, color=color)  # Histogram with optional KDE

    if show_mean:
        mean_val = np.mean(data)
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")  # Mean line

    if show_percentiles:
        for p in show_percentiles:
            val = np.percentile(data, p)
            plt.axvline(val, linestyle=':', label=f"{p}th = {val:.2f}")  # Percentile lines

    plt.xlabel(xlabel)
    plt.title(title)
    if grid:
        plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
