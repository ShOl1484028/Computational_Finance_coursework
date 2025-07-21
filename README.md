# Portfolio Analysis and Strategy Projects

This repository contains three advanced portfolio-related projects using simulated data and financial models. Each task addresses a distinct analytical perspective:

- Task 1: Portfolio optimization using Sharpe vs. Sortino ratios  
- Task 2: Trading strategy evaluation and selection (SMA vs. Bollinger Bands)  
- Task 3: Options portfolio risk analysis under hedging scenarios

> Note: All tasks involve Monte Carlo simulations or optimization routines. Execution may take 10+ minutes. Long runtime does not indicate an error.

---

## Project Overview

### Task 1: Sharpe vs. Sortino — Rethinking Strategy Metrics

This task compares portfolio optimization using the Sharpe ratio versus the Sortino ratio. It simulates efficient frontiers under both frameworks to support decision-making aligned with investor risk preferences.

### Task 2: Strategy Evaluation and Selection

This task evaluates two trading strategies—SMA crossover and Bollinger Bands—on both training and testing datasets. It applies rolling-window validation, parameter optimization, and QuantStats-based performance visualization.

### Task 3: Options Portfolio Risk Analysis

This task assesses the risk of a vanilla options portfolio. It compares unhedged and delta-hedged exposures using a Black-Scholes framework, focusing on profit and loss distributions after one year.

---

## Folder Structure

```
MANG6576_code_submission/
├── task1/
│   ├── main.py
│   ├── simulation.py
│   ├── plotter.py
│   ├── optimizer.py
│   ├── data_loader.py
│   ├── data_36304468.xlsx
│   ├── pictures/
│   └── report/
│
├── task2/
│   ├── main.py
│   ├── simulation.py
│   ├── strategy.py
│   ├── data_loader.py
│   ├── data_36304468.xlsx
│   ├── pictures/
│   └── report/
│
├── task3/
│   ├── main.py
│   ├── option_utils.py
│   ├── stats.py
│   ├── data_loader.py
│   ├── UNH.xlsx
│   ├── pictures/
│   └── report/
│
├── requirements.txt
└── README.md
```

---

## Local Installation Instructions

1. Ensure you have Python 3.8+ and pip installed.
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

**Task 1: Sharpe vs. Sortino**
```bash
cd task1
python main.py
# Expected runtime: 30–60 minutes
```

**Task 2: Strategy Evaluation and Selection**
```bash
cd task2
python main.py
# Expected runtime: 10 minutes
```

**Task 3: Options Portfolio Risk Analysis**
```bash
cd task3
python main.py
# Expected runtime: 10 minutes
```

---

## Dependencies

Listed in `requirements.txt`:

pandas==2.2.3
numpy==1.24.4
matplotlib==3.10.1
seaborn==0.13.2
yfinance==0.2.56
scipy==1.15.2
optuna==4.3.0
riskfolio==7.0.0
quantstats==0.0.62
openpyxl==3.1.5

---

## Output Access

Each task folder contains:
- `pictures/` for key plots
- `report/` for result summaries

If execution fails, you may consult these folders to interpret the results.

---

## Author Declaration

All code was developed and successfully executed in a local Python 3.8+ environment using PyCharm.  
This project has **not been tested in cloud-based environments** such as Google Colab or Jupyter Notebook,  
as execution relies on `.py` scripts and may involve long runtime due to simulation or optimization loops.  
Local execution ensures better control over dependencies, file paths, and runtime behavior.
