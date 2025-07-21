import numpy as np
import pandas as pd


def summarize_weights(w, top_n=10):
    r"""
    Summarize weight concentration and sparsity of a portfolio.

    Parameters
    ----------
    w : pd.DataFrame or pd.Series
        Portfolio weight vector.
    top_n : int, optional
        Number of top assets to aggregate. Default is 10.

    Returns
    -------
    None
        Prints top-n cumulative weight and number of non-zero positions.
    """
    weights = w.values.flatten()  # Convert to 1D array
    sorted_weights = np.sort(weights)[::-1]  # Sort in descending order
    cumulative_top = np.sum(sorted_weights[:top_n])  # Sum of top-n weights
    nonzero_count = np.sum(weights > 1e-4)  # Count weights significantly greater than zero

    print(f"Top {top_n} assets hold {cumulative_top:.2%} of the portfolio.")
    print(f"Total number of non-negligible positions: {nonzero_count}")


def load_data():
    r"""
    Load expected returns, volatilities, and correlation matrix
    from Excel file with multiple sheets.

    Returns
    -------
    expected_return : np.ndarray
        Array of expected returns in percentage format.
    volatility : np.ndarray
        Array of volatilities in percentage format.
    correlation : np.ndarray
        Correlation matrix of the assets.
    """
    data = pd.read_excel('data_36304468.xlsx', sheet_name=None, index_col=0)  # Load all sheets
    expected_return = data['Expected_Returns']['Expected Returns'].values  # 1D array
    volatility = data['Volatilities']['Volatilities'].values  # 1D array
    correlation = data['Correlation'].values  # 2D array (symmetric)
    return expected_return, volatility, correlation


def prepare_parameters():
    r"""
    Prepare daily return and risk parameters from raw annualized inputs
    for use with Riskfolio-Lib portfolio optimization.

    Returns
    -------
    Mu : pd.DataFrame
        Daily expected returns as a DataFrame.
    Vol : pd.DataFrame
        Daily volatilities as a DataFrame.
    Sigma : pd.DataFrame
        Diagonal volatility matrix.
    Cov : pd.DataFrame
        Daily covariance matrix.
    Rf : float
        Daily risk-free rate.
    Target : float
        Daily target return.
    dim : int
        Number of assets.
    """
    np.random.seed(123)  # Set seed for reproducibility
    risk_free_rate = 2  # Annual risk-free rate in percent
    target_return = 7  # Annual target return in percent

    # Load raw data from Excel
    expected_return, volatility, correlation = load_data()

    # Convert to daily decimal format
    mu = expected_return / 100 / 252  # Convert % to daily return
    vol = volatility / 100 / np.sqrt(252)  # Convert % to daily std dev
    sigma = np.diag(vol)  # Diagonal matrix of daily std dev
    cov = sigma @ correlation @ sigma  # Daily covariance matrix
    rf = risk_free_rate / 100 / 252  # Daily risk-free rate
    target = target_return / 100 / 252  # Daily target return
    dim = len(mu)  # Number of assets

    # Format outputs as DataFrames for Riskfolio compatibility
    asset_names = [f"Asset{i + 1}" for i in range(dim)]
    mu = pd.DataFrame(mu, columns=["Mu"])
    vol = pd.DataFrame(vol, columns=["Vol"])
    sigma = pd.DataFrame(sigma, index=asset_names, columns=asset_names)
    cov = pd.DataFrame(cov, index=asset_names, columns=asset_names)

    return mu, vol, sigma, cov, rf, target, dim


if __name__ == '__main__':
    # Run preparation function and print summaries
    mu, vol, sigma, cov, rf, target, dim = prepare_parameters()

    print("=== Mu (daily expected return) ===")
    print(mu.head())
    print("\n=== Vol (daily volatility) ===")
    print(vol.head())
    print("\n=== Covariance matrix (daily) ===")
    print(cov.iloc[:5, :5])  # Preview top-left 5x5 block
    print("\n=== Risk-free rate (daily):", rf)
    print("=== Target return (daily):", target)
    print("=== Number of assets:", dim)

    # Validate dimension consistency
    assert mu.shape[0] == dim
    assert cov.shape == (dim, dim)

    # Validate symmetry of covariance matrix
    assert np.allclose(cov, cov.T), "Covariance matrix is not symmetric"

    # Optional: Validate positive definiteness
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), "Covariance matrix is not positive definite"

    print("Data check passed. Format is compatible with Riskfolio-Lib.")
