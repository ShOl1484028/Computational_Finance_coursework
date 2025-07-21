import warnings
import quantstats as qs
from optimizer import *
from plotter import *

# === Suppress warnings and configure QuantStats ===
warnings.filterwarnings("ignore")
qs.extend_pandas()  # Enable QuantStats methods on Pandas objects

# === Load and split data ===
data = pd.read_excel('UNH.xlsx')  # Load historical price data
data.set_index('Date', inplace=True)  # Set date column as index
train_data, test_data = split_data(data)  # Split dataset into training and testing sets

# === SMA Strategy Optimization ===
sma_param_grid = {
    'short_window': range(5, 31),  # Short moving average window candidates
    'long_window': range(10, 61)  # Long moving average window candidates
}
sma_bounds = [
    (min(sma_param_grid['short_window']), max(sma_param_grid['short_window'])),
    (min(sma_param_grid['long_window']), max(sma_param_grid['long_window']))
]  # Define bounds for optimization

# Run restart-based optimizer for SMA parameters
sma_best_params, sma_best_sharpe = optimize_with_restarts(
    objective_func=maximize_sharpe_sma,
    train_data=train_data,
    bounds=sma_bounds,
    n_restarts=50,
    method='Nelder-Mead',
    maxiter=1000
)

# Output best parameter and performance
print(f"Final best SMA parameters: short={sma_best_params[0]}, long={int(sma_best_params[1])}")
print(f"Best Sharpe: {sma_best_sharpe:.4f}")

# Define helper to compute Sharpe for heatmap
def sharpe_lookup_sma(short, long):
    if short >= long:
        return -1e6  # Invalid combination
    return cross_validate_sma(train_data, short, long)

# Plot grid search result as heatmap
plot_sharpe_heatmap(
    score_func=sharpe_lookup_sma,
    x_range=sma_param_grid['short_window'],
    y_range=sma_param_grid['long_window'],
    best_params=sma_best_params,
    title="SMA Sharpe Heatmap",
    xlabel="Short Window",
    ylabel="Long Window",
    enforce_order=True
)

# === Optuna Optimization for SMA ===
sma_study = optimize_with_optuna('sma', train_data, n_trials=1000)
print("Best Parameters (SMA):", sma_study.best_params)
print("Best Sharpe (train):", -sma_study.best_value)

# Manually set best parameters found from study
best_short = 19
best_long = 53

# === Evaluate SMA strategy on test data ===
sma_optimized_results = sma_strategy(test_data, short_window=best_short, long_window=best_long)
optimized_returns = sma_optimized_results['Strategy_Return']  # Strategy return series
benchmark = sma_optimized_results['Log_Return']  # Buy and hold benchmark

# Compute Sharpe and return metrics
best_sharpe = optimized_returns.mean() / optimized_returns.std() * np.sqrt(252)
best_ret = optimized_returns.mean() * 252
print("Sharpe Ratio (test):", best_sharpe)

# === QuantStats Report for SMA Strategy ===
qs.reports.html(
    optimized_returns,
    benchmark=benchmark,
    output="Optimized_SMA.html",
    title="SMA Strategy vs Buy & Hold"
)

# Plot strategy signals and performance
plot_strategy_signals(
    sma_optimized_results,
    value_col='Strategy_Value',
    signal_label="SMA",
    ticker="UNH"
)

# Evaluate all grid parameter combinations on test set
result_df = evaluate_param_performance(
    test_data=test_data,
    strategy_func=sma_strategy,
    param_grid=sma_param_grid
)

# Plot Sharpe and return distribution histograms
plot_metric_percentile(result_df['sharpe'], best_sharpe, title='Sharpe Ratio Distribution (Test Set)',
                       xlabel='Sharpe Ratio')
plot_metric_percentile(result_df['annual_return'], best_ret, title='Annual Return Distribution (Test Set)',
                       xlabel='Annual Return')

# === Bollinger Band Strategy Optimization ===
boll_param_grid = {
    'window': range(5, 61),  # Lookback window for MA
    'num_std': [round(x * 0.2, 1) for x in range(5, 21)]  # Standard deviation multiplier range
}
boll_bounds = [
    (min(boll_param_grid['window']), max(boll_param_grid['window'])),
    (min(boll_param_grid['num_std']), max(boll_param_grid['num_std']))
]  # Bounds for optimization

# Run optimizer for Bollinger strategy
boll_best_params, boll_best_sharpe = optimize_with_restarts(
    objective_func=maximize_sharpe_bollinger,
    train_data=train_data,
    bounds=boll_bounds,
    n_restarts=50,
    method='Nelder-Mead',
    maxiter=1000
)

# Output best result
print(f"Best Bollinger Parameters: window={boll_best_params[0]}, num_std={boll_best_params[1]}")
print(f"Best Sharpe: {boll_best_sharpe:.4f}")

# Define lookup helper for heatmap
def sharpe_lookup_bollinger(window, num_std):
    return cross_validate_bollinger(train_data, window, num_std)

# Plot Bollinger heatmap result
plot_sharpe_heatmap(
    score_func=sharpe_lookup_bollinger,
    x_range=boll_param_grid['window'],
    y_range=boll_param_grid['num_std'],
    best_params=boll_best_params,
    title="Bollinger Sharpe Heatmap",
    xlabel="Window",
    ylabel="Num Std",
    enforce_order=False
)

# === Optuna Optimization for Bollinger ===
boll_study = optimize_with_optuna('bollinger', train_data, n_trials=1000)
print("Best Parameters (Bollinger):", boll_study.best_params)
print("Best Sharpe (train):", -boll_study.best_value)

# Set best parameters found manually
best_window = 17
best_std = 3.0

# === Evaluate Bollinger strategy on test set ===
boll_optimized_results = bollinger_strategy(test_data, window=best_window, num_std=best_std)
optimized_returns = boll_optimized_results['Strategy_Return']
benchmark = boll_optimized_results['Log_Return']

# Compute performance metrics
best_sharpe = optimized_returns.mean() / optimized_returns.std() * np.sqrt(252)
best_ret = optimized_returns.mean() * 252
print("Sharpe Ratio (test):", best_sharpe)

# Generate QuantStats HTML report
qs.reports.html(
    optimized_returns,
    benchmark=benchmark,
    output="Optimized_Bollinger.html",
    title="Bollinger Strategy vs Buy & Hold"
)

# Plot signal and equity curve
plot_strategy_signals(
    strategy_data=boll_optimized_results,
    value_col='Strategy_Value',
    signal_label="Bollinger",
    ticker="UNH"
)

# Evaluate all parameter combinations on test set
result_df = evaluate_param_performance(
    test_data=test_data,
    strategy_func=bollinger_strategy,
    param_grid=boll_param_grid
)

# Plot result distributions
plot_metric_percentile(result_df['sharpe'], best_sharpe, title='Sharpe Ratio Distribution (Test Set)',
                       xlabel='Sharpe Ratio')
plot_metric_percentile(result_df['annual_return'], best_ret, title='Annual Return Distribution (Test Set)',
                       xlabel='Annual Return')
