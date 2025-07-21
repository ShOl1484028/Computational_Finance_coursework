import matplotlib.pyplot as plt
import seaborn as sns


def plot_return_distribution(returns, title='Simulated Final Returns of Optimal Portfolio', color='steelblue'):
    r"""
    Plot the distribution of annual returns using histogram and KDE curve.

    Parameters
    ----------
    returns : array-like or pd.Series
        One-dimensional array or Series containing annualized returns.
    title : str, optional
        Title of the plot. Default is 'Simulated Final Returns of Optimal Portfolio'.
    color : str, optional
        Color used for the histogram and KDE curve. Default is 'steelblue'.

    Returns
    -------
    None
        Displays the plot.
    """
    plt.figure(figsize=(10, 6))  # Set figure size

    # Plot histogram and KDE (Kernel Density Estimate)
    sns.histplot(returns, bins=50, kde=True, color=color, stat='density', alpha=0.6)

    # Plot mean return line
    mean_ret = returns.mean()
    plt.axvline(mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.2%}')  # Vertical mean line
    plt.text(mean_ret, plt.ylim()[1] * 0.9, f'{mean_ret:.2%}', color='red', ha='right')  # Annotate mean

    # Set plot labels and formatting
    plt.title(title)
    plt.xlabel('Annual Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
