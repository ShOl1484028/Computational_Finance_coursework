from strategy import *
from scipy.optimize import minimize
import optuna
import numpy as np


def split_data(data, train_ratio=0.8):
    r"""
    Split dataset into training and testing sets.

    Parameters
    ----------
    data : pd.DataFrame
        Full time series dataset.
    train_ratio : float
        Fraction of data to use for training.

    Returns
    -------
    tuple
        (train_data, test_data)
    """
    split_index = int(len(data) * train_ratio)  # Determine split index
    return data.iloc[:split_index], data.iloc[split_index:]  # Slice into train and test sets


def cross_validate_sma(train_data, short_window, long_window, train_size=1000, val_size=500):
    r"""
    Perform rolling-window cross-validation for the SMA strategy.

    Parameters
    ----------
    train_data : pd.DataFrame
        Input time series data for training.
    short_window : int
        Short-term moving average window.
    long_window : int
        Long-term moving average window.
    train_size : int, optional
        Size of training window.
    val_size : int, optional
        Size of validation window.

    Returns
    -------
    float
        Median Sharpe ratio across all validation windows.
    """
    total_size = len(train_data)
    scores = []  # Store Sharpe scores
    train_start = 0

    while True:
        train_end = train_start + train_size
        val_end = train_end + val_size
        if val_end > total_size:
            break  # Exit if validation window exceeds total data size

        # Prepare data slice and validation index
        full_data = train_data.iloc[train_start:val_end]
        val_index = train_data.iloc[train_end:val_end].index

        try:
            result = sma_strategy(full_data, short_window, long_window)
            val_result = result[result.index.isin(val_index)]
            rets = val_result['Strategy_Return']
            sharpe = rets.sharpe()
            if np.isfinite(sharpe):
                scores.append(sharpe)  # Append valid Sharpe ratio
        except:
            pass  # Ignore errors in strategy execution

        train_start += val_size  # Advance window

    return np.median(scores) if scores else -1e6  # Penalize if no valid scores


def maximize_sharpe_sma(params, train_data):
    r"""
    Objective function for SMA optimization.

    Parameters
    ----------
    params : list
        [short_window, long_window]
    train_data : pd.DataFrame
        Training data.

    Returns
    -------
    float
        Negative Sharpe score (for minimization).
    """
    short = int(params[0])
    long = int(params[1])

    # Check if parameters are within acceptable bounds
    if not (5 <= short <= 30 and 10 <= long <= 60 and short <= long):
        return 1e6

    try:
        score = cross_validate_sma(train_data, short, long)
        return -score if np.isfinite(score) else 1e6  # Return negative Sharpe for minimization
    except:
        return 1e6  # Return penalty if any error


def cross_validate_bollinger(train_data, window, num_std, train_size=1000, val_size=500):
    r"""
    Perform rolling-window cross-validation for the Bollinger Band strategy.

    Parameters
    ----------
    train_data : pd.DataFrame
        Input dataset.
    window : int
        Moving average window.
    num_std : float
        Number of standard deviations for the bands.
    train_size : int
        Training window size.
    val_size : int
        Validation window size.

    Returns
    -------
    float
        Median Sharpe ratio over all windows.
    """
    total_size = len(train_data)
    scores = []  # Store Sharpe ratios
    train_start = 0

    while True:
        train_end = train_start + train_size
        val_end = train_end + val_size
        if val_end > total_size:
            break  # End of data reached

        full_data = train_data.iloc[train_start:val_end]
        val_index = train_data.iloc[train_end:val_end].index

        try:
            result = bollinger_strategy(full_data, window, num_std)
            val_result = result[result.index.isin(val_index)]
            rets = val_result['Strategy_Return']
            sharpe = rets.sharpe()
            if np.isfinite(sharpe):
                scores.append(sharpe)
        except:
            pass  # Skip invalid result

        train_start += val_size

    return np.median(scores) if scores else -1e6


def maximize_sharpe_bollinger(params, train_data):
    r"""
    Objective function for Bollinger Band optimization.

    Parameters
    ----------
    params : list
        [window, num_std]
    train_data : pd.DataFrame
        Input dataset for training.

    Returns
    -------
    float
        Negative Sharpe score for optimizer.
    """
    window = int(params[0])
    num_std = float(params[1])

    if not (5 <= window <= 60 and 1.0 <= num_std <= 4.0):
        return 1e6  # Reject parameters out of range

    try:
        score = cross_validate_bollinger(train_data, window, num_std)
        return -score if np.isfinite(score) else 1e6
    except:
        return 1e6  # Handle unexpected failure


def optimize_with_restarts(objective_func, train_data, bounds, n_restarts=50, method='Nelder-Mead', maxiter=1000):
    r"""
    Perform random-restart local optimization.

    Parameters
    ----------
    objective_func : callable
        Function to minimize.
    train_data : pd.DataFrame
        Input training data.
    bounds : list of tuples
        Bounds for each parameter.
    n_restarts : int
        Number of random restarts.
    method : str
        Optimization method (e.g., 'Nelder-Mead').
    maxiter : int
        Maximum iterations per run.

    Returns
    -------
    tuple
        (best_params, best_score)
    """
    best_score = -1e9  # Best score initialized low
    best_params = None

    # Generate multiple random starting points within bounds
    initial_list = [
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_restarts)
    ]

    for x0 in initial_list:
        try:
            res = minimize(lambda x: objective_func(x, train_data),
                           x0=x0,
                           method=method,
                           options={'maxiter': maxiter})
            score = -res.fun  # Convert from loss to score
            if score > best_score:
                best_score = score
                best_params = res.x  # Update best result
        except:
            continue  # Skip failed runs

    # Final parameter formatting for readability
    formatted_params = [
        int(round(p)) if i == 0 else round(p, 2)
        for i, p in enumerate(best_params)
    ]

    return formatted_params, best_score  # Return optimal configuration


def optimize_with_optuna(strategy_type, train_data, n_trials=100):
    r"""
    Use Optuna to find optimal strategy parameters.

    Parameters
    ----------
    strategy_type : str
        'sma' or 'bollinger'
    train_data : pd.DataFrame
        Data used for training and evaluation.
    n_trials : int
        Number of Optuna trials.

    Returns
    -------
    optuna.study.Study
        Optimized study object with best parameters.
    """
    def objective(trial):
        # Define parameter space based on strategy type
        if strategy_type == 'sma':
            short = trial.suggest_int('short', 5, 31)
            long = trial.suggest_int('long', short + 1, 61)
            return maximize_sharpe_sma([short, long], train_data)
        elif strategy_type == 'bollinger':
            window = trial.suggest_int('window', 5, 60)
            num_std = trial.suggest_float('num_std', 1.0, 4.0)
            return maximize_sharpe_bollinger([window, num_std], train_data)
        else:
            raise ValueError("Unsupported strategy type. Use 'sma' or 'bollinger'.")

    study = optuna.create_study(direction='minimize')  # We minimize negative Sharpe ratio
    study.optimize(objective, n_trials=n_trials)  # Run optimization loop
    return study  # Return the result study object
