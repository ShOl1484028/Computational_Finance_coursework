from simulation import *
from plotter import *
from data_loader import *
import riskfolio as rp
import warnings
import matplotlib.pyplot as plt

# === Setup ===
warnings.filterwarnings("ignore")  # Suppress warnings
pd.options.display.float_format = '{:.4%}'.format  # Format floats as percentages

# === Load and prepare data ===
mu, vol, sigma, cov, rf, target, dim = prepare_parameters()
simulated_ret_df = generate_asset_returns(mu, cov)  # Simulate daily returns from mu and cov

# === Initialize Riskfolio Portfolio object ===
port = rp.Portfolio(returns=simulated_ret_df)  # Pass returns into the optimizer

# Annualized values for optimization
port.mu = (mu * 252 * 100).T  # Convert mu to annualized %
port.cov = cov * 252 * 10000  # Convert cov to annualized %^2
rf_for_opt = rf * 252 * 100  # Convert risk-free rate to annualized %

# Daily values for plotting and evaluation
mu_plot = mu.T
cov_plot = cov
rf_plot = rf

# === Sharpe Optimization ===
w_mv = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=rf_for_opt, l=0, hist=True)  # Max Sharpe
mv_frontier = port.efficient_frontier(model='Classic', rm='MV', points=50, rf=rf, hist=True)  # Efficient frontier

# Plot efficient frontier and Sharpe-optimal portfolio
fig, ax = plt.subplots(figsize=(10, 6))
rp.plot_frontier(
    w_frontier=mv_frontier,
    mu=mu_plot,
    cov=cov_plot,
    returns=simulated_ret_df,
    rm='MV',
    rf=rf_plot,
    t_factor=252,
    label='Max Sharpe',
    marker='*',
    s=16,
    c='r',
    w=w_mv,
    ax=ax,
)
figures = plt.gcf()
ax0 = figures.axes[1]  # Second subplot for annotations

# Compute Sharpe-optimal portfolio metrics
risk_mv = rp.Sharpe_Risk(returns=simulated_ret_df, w=w_mv, rm='MV', rf=rf_plot) * np.sqrt(252)  # Annualized std
ret_mv = (mu_plot @ w_mv.values).iloc[0, 0] * 252  # Annualized return
sharpe_mv = (ret_mv - rf * 252) / risk_mv  # Sharpe ratio

# Annotate optimal point
ax.scatter(risk_mv, ret_mv, color='red', s=100, marker='*')
ax0.annotate(
    f"({risk_mv:.2%}, {ret_mv:.2%})",
    xy=(risk_mv, ret_mv),
    xycoords='data',
    textcoords='offset points',
    xytext=(10, 0),
    fontsize=9,
    color='black',
    ha='left',
    va='center'
)
plt.show()

# Print Sharpe-optimal results
print("\n=== Sharpe-optimal Portfolio Metrics ===")
print(f"Annual Return: {ret_mv:.2%}")
print(f"Annual Volatility: {risk_mv:.2%}")
print(f"Sharpe Ratio: {sharpe_mv:.2f}")
print("\n=== Optimal Weights ===")
print(w_mv.T.to_string(index=False))
print("\n=== Weight Concentration (Sharpe-optimal) ===")
summarize_weights(w_mv)  # Show top holdings

# Simulate final return distribution of Sharpe-optimal portfolio
sigma_mv = np.sqrt((risk_mv ** 2) / 252)  # Daily std from annual volatility
simulated_opt_log_returns = simulate_optimal_portfolio_returns(mu=ret_mv / 252, sigma=sigma_mv)  # Simulate paths
cumulative_log_returns = simulated_opt_log_returns.sum(axis=1)  # Sum log returns
final_simple_returns = np.exp(cumulative_log_returns) - 1  # Convert to simple returns

plot_return_distribution(final_simple_returns)  # Show histogram of simulated returns

# === Sortino Optimization ===
w_sortino = port.optimization(model='Classic', rm='SLPM', obj='Sharpe', rf=target * 100, l=0, hist=True)  # Max Sortino
sortino_frontier = port.efficient_frontier(model='Classic', rm='SLPM', points=50, rf=target, hist=True)  # Frontier

# Plot efficient frontier and Sortino-optimal portfolio
fig, ax = plt.subplots(figsize=(10, 6))
rp.plot_frontier(
    w_frontier=sortino_frontier,
    mu=mu_plot,
    cov=cov_plot,
    returns=simulated_ret_df,
    rm='SLPM',
    rf=target,
    t_factor=252,
    label='Max Sortino',
    marker='*',
    s=16,
    c='r',
    w=w_sortino,
    ax=ax,
)
figures = plt.gcf()
ax0 = figures.axes[1]  # Second axis for annotation

# Compute Sortino-optimal portfolio metrics
risk_sortino = rp.Sharpe_Risk(returns=simulated_ret_df, w=w_sortino, rm='SLPM', rf=target) * np.sqrt(
    252)  # Downside risk
ret_sortino = (mu_plot @ w_sortino.values).iloc[0, 0] * 252  # Annualized return
sortino_ratio = (ret_sortino - target * 252) / risk_sortino  # Sortino ratio

# Annotate optimal Sortino point
ax.scatter(risk_sortino, ret_sortino, color='red', s=100, marker='*')  # Red star on plot
ax0.annotate(
    f"({risk_sortino:.2%}, {ret_sortino:.2%})",
    xy=(risk_sortino, ret_sortino),
    xycoords='data',
    textcoords='offset points',
    xytext=(10, 0),
    fontsize=9,
    color='black',
    ha='left',
    va='center'
)
plt.show()

# Print Sortino-optimal results
print("\n=== Sortino-optimal Portfolio Metrics ===")
print(f"Annual Return: {ret_sortino:.2%}")
print(f"Annual Volatility: {risk_sortino:.2%}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print("\n=== Optimal Weights ===")
print(w_sortino.T.to_string(index=False))
print("\n=== Weight Concentration (Sortino-optimal) ===")
summarize_weights(w_sortino)  # Summary of position concentration

# Simulate return distribution of Sortino-optimal portfolio
sigma_srt = np.sqrt((risk_sortino ** 2) / 252)  # Convert to daily std
target_daily_ret = ret_sortino / 252
sim_log_ret_srt = simulate_optimal_portfolio_returns(mu=target_daily_ret, sigma=sigma_srt)  # Simulate returns
cum_log_ret_srt = sim_log_ret_srt.sum(axis=1)  # Cumulative log returns
final_ret_srt = np.exp(cum_log_ret_srt) - 1  # Convert to simple return

plot_return_distribution(
    returns=final_ret_srt,
    title='Simulated Final Returns of Sortino-Optimal Portfolio'  # Annotated plot title
)
