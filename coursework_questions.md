# Coursework Questions

## 1. Portfolio Management and Risk Measures

You are a portfolio manager for an asset management firm, choosing your position from a universe of 1,000 stocks. The daily returns of these stocks are modeled as normal random variables, with expected returns and covariance matrix specified in the personalized datasets. The continuously compounded risk-free rate of return is 2% per annum, and you may assume there are 252 trading days per year.

Your boss measures your portfolio's performance using standard deviation and the Sharpe ratio, but is considering switching to using downside risk (relative to a target return of the risk-free rate plus 5%), and the Sortino ratio.

**Write Python scripts to perform the following tasks:**
- Find the Markowitz (mean-variance) optimal portfolio weights.
- Plot a scattergraph of the risk-return characteristics of some sample portfolios, along with the optimal portfolio.
- Simulate (N=10,000) the return of the optimal portfolio, and illustrate with histograms.
- Repeat the above steps with the risk-measure switched from standard deviation to downside risk (that is, standard deviation of losses below the benchmark).

**Using the above results, write a report for your boss arguing for or against the policy change. The report should be structured in the following sections:**
- 1.1 Executive Summary [~100 words]: a concise summary of your main findings, including a policy recommendation and brief explanation of your reasoning.
- 1.2 Background [~300 words]: explain the problem under consideration, including the theoretical differences between the two risk measures.
- 1.3 Optimization [~300 words]: explain how your optimization code works, at a level suitable for a financial executive who may not have a computational background.
- 1.4 Comparison of optimal returns [~300 words]: discuss your scatterplots and histograms, and use them to argue for your policy proposal. For additional credit, discuss the robustness of your findings to changes in the parameters.

---

## 2. Algorithmic Trading Strategy Development and Comparison

You are a trader at a hedge fund, attempting to find a profitable trading strategy. Personalized training and out-of-sample datasets for closing prices of the asset to be traded have been provided. Choose at least two candidates for your trading strategy to compare, based on the material in the lectures and/or by researching the literature – e.g., technical analysis strategies, time series methods such as ARIMA, machine learning methods, etc.

**Write Python scripts to perform the following tasks:**
- Train your candidate strategies using the training data to select optimal parameters, including cross-validation if appropriate.
- Evaluate the performance of your strategies using the out-of-sample data.
- Plot the performance of your strategy in an appropriate way.

**Using the above results, write a report discussing the relative merits of implementing either strategy. The report should be structured in the following sections:**
- 2.1 Executive Summary [~100 words]: a concise summary of your main findings, including a policy recommendation and brief explanation of your reasoning.
- 2.2 Background [~300 words]: explain the problem under consideration, including the theoretical differences between the two strategies.
- 2.3 Optimization [~300 words]: explain how your training and evaluation code works, at a level suitable for a financial executive who may not have a computational background.
- 2.4 Comparison of strategies [~300 words]: discuss your findings, and their robustness to changes in the parameters.

---

## 3. Derivatives Portfolio Risk Management

You are a derivatives trader, running a book of vanilla options on shares of a company. The details of your portfolio position are given in the personalized dataset. The spot price of the shares follows the Black-Scholes model, with parameters: drift mu = 10% per annum, volatility sigma = 35% per annum, and its value at time zero is S0 = $100. The continuously compounded risk-free interest rate is 5% per annum, and you may assume there are 252 trading days a year.

**Write a Python script to perform the following tasks:**
- Simulate the spot price at close of trading of a share, at daily intervals, over the next year.
- Simulate the value at close of trading of your portfolio, at daily intervals over the next year, using the Black-Scholes pricing formulae.
- Simulate the delta of your portfolio, at daily intervals over the next year, using the Black-Scholes delta formulae.
- Plot a histogram of the distribution of the profit and loss over the next year of (i) the unhedged portfolio and (ii) the daily delta-hedged portfolio.
- Find the one-year 99% value-at-risk, of (i) the unhedged portfolio and (ii) the delta-hedged portfolio.

**Using the above results, write a risk management report of your portfolio. The report should be structured in the following sections:**
- 3.1 Executive Summary [~100 words]: a concise summary of your main findings.
- 3.2 Background [~300 words]: explain the details of your portfolio, including an explanation and description of all contracts, and their parameters.
- 3.3 Simulation [~300 words]: explain how your simulation code works, at a level suitable for a financial executive who may not have a computational background.
- 3.4 Risk evaluation [~300 words]: discuss your histograms, and the efficacy of delta-hedging. 