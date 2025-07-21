import numpy as np
import pandas as pd


def generate_weights(n_portfolios, n_assets):
    r"""
    Generate normalized positive weights for multiple portfolios.

    Parameters
    ----------
    n_portfolios : int
        Number of portfolios to simulate.
    n_assets : int
        Number of assets in each portfolio.

    Returns
    -------
    weights : np.ndarray
        Weight matrix of shape (n_portfolios, n_assets) where each row sums to 1.
    """
    raw = np.random.rand(n_portfolios, n_assets)  # Generate random weights
    weights = raw / raw.sum(axis=1, keepdims=True)  # Normalize to sum to 1
    return weights


def generate_asset_returns(mu, cov, n_samples=10000):
    r"""
    Generate daily asset return samples using a multivariate normal distribution.

    Parameters
    ----------
    mu : array-like
        Daily expected returns (decimal format).
    cov : array-like
        Daily covariance matrix.
    n_samples : int, optional
        Number of simulated time steps. Default is 5000.

    Returns
    -------
    pd.DataFrame
        Simulated return matrix with asset names as columns.
    """
    np.random.seed(123)  # Set random seed for reproducibility
    mu = np.asarray(mu).flatten()
    cov = np.asarray(cov)

    simulated = np.random.multivariate_normal(mean=mu, cov=cov, size=n_samples)  # Simulate returns
    n_assets = len(mu)
    columns = [f'Asset{i+1}' for i in range(n_assets)]  # Asset labels
    simulated_df = pd.DataFrame(simulated, columns=columns)
    return simulated_df


def simulate_optimal_portfolio_returns(mu, sigma, n_days=252, n_paths=10000):
    r"""
    Simulate log return paths for an optimal portfolio using normal distribution.

    Parameters
    ----------
    mu : float
        Mean daily log return.
    sigma : float
        Standard deviation of daily log return.
    n_days : int, optional
        Number of days per path. Default is 252 (1 year).
    n_paths : int, optional
        Number of simulation paths. Default is 10000.

    Returns
    -------
    np.ndarray
        Simulated return matrix of shape (n_paths, n_days).
    """
    return np.random.normal(loc=mu, scale=sigma, size=(n_paths, n_days))  # Matrix of simulated log returns

