import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, T, r, vol):
    r"""
    Calculate the price of a European call option using the Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Current price of the underlying asset.
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
        Price of the European call option.
    """
    L = np.log(S0 / K) + (r + 0.5 * vol ** 2) * T  # Numerator of d1
    V = vol * np.sqrt(T)  # Denominator of d1, and used in d2
    d1 = L / V
    d2 = d1 - V

    N = norm.cdf  # Cumulative distribution function for standard normal
    C = S0 * N(d1) - K * np.exp(-r * T) * N(d2)  # Call option formula
    return C


def black_scholes_put(S0, K, T, r, vol):
    r"""
    Calculate the price of a European put option using the Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Current price of the underlying asset.
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
        Price of the European put option.
    """
    L = np.log(S0 / K) + (r + 0.5 * vol ** 2) * T  # Numerator of d1
    V = vol * np.sqrt(T)  # Denominator of d1, and used in d2
    d1 = L / V
    d2 = d1 - V

    N = norm.cdf  # Cumulative distribution function for standard normal
    P = K * np.exp(-r * T) * N(-d2) - S0 * N(-d1)  # Put option formula
    return P


def bs_risk_reversal_price(S0, K1, K2, T, r, vol):
    r"""
    Calculate the price of a risk reversal position: long call and short put.

    Parameters
    ----------
    S0 : float
        Current price of the underlying asset.
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
        Net price of the risk reversal position.
    """
    call_price = black_scholes_call(S0, K2, T, r, vol)  # Long call
    put_price = black_scholes_put(S0, K1, T, r, vol)  # Short put
    return call_price - put_price
