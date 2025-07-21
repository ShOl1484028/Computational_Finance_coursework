import numpy as np
from scipy.stats import norm


def bs_call_delta(S0, K, T, r, vol):
    r"""
    Compute the Black-Scholes delta for a European call option.

    Parameters
    ----------
    S0 : float
        Current underlying asset price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        Delta of the call option. If T <= 0, returns 0.
    """
    if T <= 0:
        return 0.0

    # Compute d1 using Black-Scholes formula
    L = np.log(S0 / K) + (r + 0.5 * vol ** 2) * T
    V = vol * np.sqrt(T)
    d1 = L / V

    N = norm.cdf  # Standard normal CDF
    delta = N(d1)
    return delta


def bs_put_delta(S0, K, T, r, vol):
    r"""
    Compute the Black-Scholes delta for a European put option.

    Parameters
    ----------
    S0 : float
        Current underlying asset price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        Delta of the put option. If T <= 0, returns 0.
    """
    if T <= 0:
        return 0.0

    # Compute d1 using Black-Scholes formula
    L = np.log(S0 / K) + (r + 0.5 * vol ** 2) * T
    V = vol * np.sqrt(T)
    d1 = L / V

    N = norm.cdf  # Standard normal CDF
    delta = N(d1) - 1  # Put delta formula
    return delta


def bs_risk_reversal_delta(S0, K1, K2, T, r, vol):
    r"""
    Compute the delta of a risk reversal position: long call and short put.

    Parameters
    ----------
    S0 : float
        Current underlying asset price.
    K1 : float
        Strike price of the put option (short leg).
    K2 : float
        Strike price of the call option (long leg).
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        Net delta of the risk reversal position.
    """
    if T <= 0:
        return 0.0

    # Long call delta (positive)
    call_delta = bs_call_delta(S0, K2, T, r, vol)

    # Short put delta (negative, subtracting a negative)
    put_delta = bs_put_delta(S0, K1, T, r, vol)

    return call_delta - put_delta
