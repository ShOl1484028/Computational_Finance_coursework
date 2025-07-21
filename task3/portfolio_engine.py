from bs_pricing import *
from bs_greeks import *
import numpy as np


def rr_portfolio_value_paths(spot_paths, K1, K2, maturity1, maturity2, rf, vol, dt, weight1=5, weight2=-3):
    r"""
    Simulate the value paths of a portfolio composed of two risk reversal positions.

    Parameters
    ----------
    spot_paths : np.ndarray
        Array of shape (n_paths, T+1) representing simulated spot prices.
    K1 : float
        Strike price of the short put.
    K2 : float
        Strike price of the long call.
    maturity1 : float
        Time to maturity for the first position (in years).
    maturity2 : float
        Time to maturity for the second position (in years).
    rf : float
        Risk-free rate.
    vol : float
        Volatility of the underlying asset.
    dt : float
        Time step size (e.g., 1/252 for daily steps).
    weight1 : float, optional
        Weight of the first position. Default is 5.
    weight2 : float, optional
        Weight of the second position. Default is -3.

    Returns
    -------
    portfolio_paths : np.ndarray
        Simulated portfolio value paths, shape (n_paths, T+1).
    initial_value : float
        Initial portfolio value.
    """
    n_paths, steps = spot_paths.shape
    rr1 = np.zeros_like(spot_paths)
    rr2 = np.zeros_like(spot_paths)

    for path in range(n_paths):
        for t in range(steps):
            tau1 = max(maturity1 - t * dt, 0)  # Time to maturity for first leg
            tau2 = max(maturity2 - t * dt, 0)  # Time to maturity for second leg
            S = spot_paths[path, t]
            rr1[path, t] = bs_risk_reversal_price(S, K1, K2, tau1, rf, vol)
            rr2[path, t] = bs_risk_reversal_price(S, K1, K2, tau2, rf, vol)

    # Compute initial value at t=0 based on spot_paths[0,0]
    initial_val = (weight1 * bs_risk_reversal_price(spot_paths[0, 0], K1, K2, maturity1, rf, vol) +
                   weight2 * bs_risk_reversal_price(spot_paths[0, 0], K1, K2, maturity2, rf, vol))

    portfolio = weight1 * rr1 + weight2 * rr2  # Combine positions
    return portfolio, initial_val


def rr_portfolio_delta_paths(spot_paths, K1, K2, maturity1, maturity2, rf, vol, dt, weight1=5, weight2=-3):
    r"""
    Simulate the daily delta exposure of a portfolio with two risk reversal positions.

    Parameters
    ----------
    spot_paths : np.ndarray
        Array of shape (n_paths, T+1) representing spot price paths.
    K1 : float
        Strike of the short put leg.
    K2 : float
        Strike of the long call leg.
    maturity1 : float
        Maturity of the first structure.
    maturity2 : float
        Maturity of the second structure.
    rf : float
        Risk-free rate.
    vol : float
        Annual volatility.
    dt : float
        Time step size.
    weight1 : float, optional
        Position weight of the first risk reversal.
    weight2 : float, optional
        Position weight of the second risk reversal.

    Returns
    -------
    delta_paths : np.ndarray
        Simulated delta exposures, shape (n_paths, T+1).
    """
    n_paths, T_plus_1 = spot_paths.shape
    delta_paths = np.zeros_like(spot_paths)

    for path in range(n_paths):
        for t in range(T_plus_1):
            S_t = spot_paths[path, t]
            tau1 = max(maturity1 - t * dt, 0)
            tau2 = max(maturity2 - t * dt, 0)

            delta1 = bs_risk_reversal_delta(S_t, K1, K2, tau1, rf, vol)
            delta2 = bs_risk_reversal_delta(S_t, K1, K2, tau2, rf, vol)

            delta_paths[path, t] = weight1 * delta1 + weight2 * delta2

    return delta_paths


def summarize_strategy_performance(spot_final, pnl_unhedged, pnl_hedged, var_unhedged, var_hedged, label=""):
    r"""
    Print summary statistics of strategy performance.

    Parameters
    ----------
    spot_final : np.ndarray
        Final spot prices across all paths.
    pnl_unhedged : np.ndarray
        PnL results of the unhedged strategy.
    pnl_hedged : np.ndarray
        PnL results of the delta-hedged strategy.
    var_unhedged : float
        Value-at-Risk (99%) of the unhedged portfolio.
    var_hedged : float
        Value-at-Risk (99%) of the delta-hedged portfolio.
    label : str, optional
        Optional label to prefix the output.

    Returns
    -------
    None
        Prints the results to stdout.
    """
    print(f"\n===== Performance Summary: {label} =====")
    print(f"Spot Mean: {np.mean(spot_final):.2f}")
    print(f"Spot Std: {np.std(spot_final):.2f}")
    print(f"Unhedged Mean: {np.mean(pnl_unhedged):.2f}")
    print(f"Unhedged Std: {np.std(pnl_unhedged):.2f}")
    print(f"Hedged Mean: {np.mean(pnl_hedged):.2f}")
    print(f"Hedged Std: {np.std(pnl_hedged):.2f}")
    print(f"99% VaR (Unhedged): {var_unhedged:.2f}")
    print(f"99% VaR (Delta-Hedged): {var_hedged:.2f}")
