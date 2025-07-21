import numpy as np


def simulate_return(g, sigma, dt, z):
    r"""
    Compute log return using geometric Brownian motion formula.

    Parameters
    ----------
    g : float
        Drift term.
    sigma : float
        Volatility of the asset.
    dt : float
        Time step size.
    z : float or np.ndarray
        Random standard normal draw(s).

    Returns
    -------
    float or np.ndarray
        Log return(s) over the given step(s).
    """
    log_ret = (g - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return log_ret


def simulate_price_step(S0, g, sigma, dt, z):
    r"""
    Simulate the next asset price from current price using log return.

    Parameters
    ----------
    S0 : float
        Current asset price.
    g : float
        Drift term.
    sigma : float
        Volatility.
    dt : float
        Time increment.
    z : float
        Standard normal random variable.

    Returns
    -------
    float
        Next simulated asset price.
    """
    log_ret = simulate_return(g, sigma, dt, z)
    Sn = np.exp(log_ret) * S0  # Apply log return to get new price
    return Sn


def cumulative_log_returns(g, sigma, dt, z):
    r"""
    Compute cumulative log returns for one or multiple paths.

    Parameters
    ----------
    g : float
        Drift term.
    sigma : float
        Volatility.
    dt : float
        Time step.
    z : np.ndarray
        Standard normal draws (1D for single path, 2D for multiple).

    Returns
    -------
    np.ndarray
        Cumulative log returns starting from zero.
    """
    if z.ndim == 1:
        steps = len(z)
        log_ret_series = np.zeros(steps + 1)
        log_ret_series[1:] = simulate_return(g, sigma, dt, z)  # Compute returns
        cum_log_ret = np.cumsum(log_ret_series)  # Accumulate log returns
    elif z.ndim == 2:
        paths, steps = z.shape
        log_ret_matrix = np.zeros((paths, steps + 1))
        log_ret_matrix[:, 1:] = simulate_return(g, sigma, dt, z)
        cum_log_ret = np.cumsum(log_ret_matrix, axis=1)  # Accumulate along steps
    else:
        raise ValueError("z must be a 1D or 2D array.")

    return cum_log_ret
