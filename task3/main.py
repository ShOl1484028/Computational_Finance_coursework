from sim_paths import *
from plotter import *
from portfolio_engine import *
import numpy as np

# === Simulation Parameters ===
np.random.seed(123)
n_paths = 10000
horizon = 1
maturity1 = 1.5
maturity2 = 3
freq = 252
timesteps = horizon * freq
dt = 1 / 252
initial_spot = 100
mu = 0.1
rf = 0.05
vol = 0.35

# === Simulate Paths ===
Z1 = np.random.normal(0, 1, size=timesteps)  # Single path noise
Z2 = np.random.normal(0, 1, size=(n_paths, timesteps))  # Multi-path noise
K1 = 68.1341258851323
K2 = 129.61866873647

# Single path log returns and price
cum_log_ret1 = cumulative_log_returns(mu, vol, dt, Z1)
price_series1 = initial_spot * np.exp(cum_log_ret1)
yearly_log_ret1 = cum_log_ret1[-1]
yearly_price1 = price_series1[-1]

# Multiple path log returns and price
cum_log_ret2 = cumulative_log_returns(mu, vol, dt, Z2)
price_series2 = initial_spot * np.exp(cum_log_ret2)
yearly_log_ret2 = cum_log_ret2[:, -1]
yearly_price2 = price_series2[:, -1]

# Time axis
t_axis = np.linspace(0, horizon, timesteps + 1)

# === Plot Simulated Paths ===
plot_single_path(t_axis, price_series1)
plot_multiple_paths(t_axis, price_series2)

# === Simulate Portfolio Value Paths ===
port_paths, initial_val = rr_portfolio_value_paths(price_series2, K1, K2, maturity1, maturity2, rf, vol, dt)
pnl_paths = port_paths - initial_val  # Unhedged PnL paths

plot_multiple_paths(
    np.linspace(0, 1, port_paths.shape[1]),
    port_paths,
    title="Simulated Portfolio Value Paths Over 1 Year",
    ylabel="Spot Price"
)

# === Simulate Delta Paths ===
delta_paths = rr_portfolio_delta_paths(price_series2, K1, K2, maturity1, maturity2, rf, vol, dt)
t_axis = np.linspace(0, horizon, delta_paths.shape[1])
plot_multiple_paths(t_axis, delta_paths, title="Simulated Daily Delta of Portfolio", ylabel="Delta")

# === Delta Hedging ===
dS = price_series2[:, 1:] - price_series2[:, :-1]  # Spot price differences
delta_for_hedge = delta_paths[:, :-1]  # Delta used for hedging
hedge_income_daily = delta_for_hedge * dS  # Daily hedge gains
hedge_income_cumsum = np.cumsum(hedge_income_daily, axis=1)  # Accumulated gains

# Add column of zeros to align with portfolio path shape
zero_column = np.zeros((hedge_income_cumsum.shape[0], 1))
hedge_income_cumsum_full = np.hstack([zero_column, hedge_income_cumsum])

# === Compute Hedged PnL Paths ===
portfolio_paths = pnl_paths + initial_val  # Reconstruct total value
pnl_unhedged = pnl_paths[:, -1]
var_unhedged = np.percentile(pnl_unhedged, 1)  # 99% VaR (unhedged)

hedged_portfolio_paths = portfolio_paths - hedge_income_cumsum_full
pnl_hedged = hedged_portfolio_paths[:, -1]  # Hedged final PnL
var_hedged = np.percentile(pnl_hedged, 1)  # 99% VaR (hedged)

# === Plot Hedged PnL Paths ===
t_axis = np.linspace(0, horizon, hedged_portfolio_paths.shape[1])
plot_multiple_paths(
    t_axis,
    hedged_portfolio_paths,
    title="Delta-Hedged Portfolio PnL Paths Over 1 Year",
    ylabel="Delta-Hedged PnL"
)

# === Plot Distributions ===
plot_distribution(
    data=price_series2[:, -1],
    xlabel="Spot Price",
    title="Simulated Spot Price Distribution"
)
plot_distribution(
    data=pnl_unhedged,
    xlabel="Profit and Loss at Year End",
    title="Unhedged Portfolio PnL Distribution"
)
plot_distribution(
    data=pnl_hedged,
    xlabel="Profit and Loss at Year End (Delta Hedged)",
    title="Delta-Hedged Portfolio PnL Distribution"
)

# === Output Performance Summary ===
summarize_strategy_performance(
    spot_final=price_series2[:, -1],
    pnl_unhedged=pnl_unhedged,
    pnl_hedged=pnl_hedged,
    var_unhedged=var_unhedged,
    var_hedged=var_hedged,
    label="Risk Reversal Portfolio"
)
