Portfolio Tracker (Python)

A simple, server-friendly portfolio tracker built in Python.
Tracks Ethereum (ETH), Gold (GLD ETF), and Nvidia (NVDA) by default, but can be extended to any asset supported by yfinance.

✨ Features

Downloads daily price data from Yahoo Finance

Calculates:

Annualized return

Annualized volatility

Sharpe ratio

Maximum drawdown

Portfolio aggregation (based on user-defined weights)

Correlation analysis between assets

Monte Carlo simulation (5-year terminal values)

Saves plots to PNG (no GUI required):

Normalized prices

Portfolio cumulative growth

Rolling correlation (ETH vs NVDA)



Output

When you run the script, it creates an outputs/ folder in the project directory with charts:

outputs/
├── normalized_prices.png
├── portfolio_cumulative_growth.png
└── rolling_corr_ETH_NVDA.png


Getting Started
1. Clone the repo

git clone https://github.com/0xstring/portfolio-tracker.git

cd portfolio-tracker


2. Create a virtual environment (recommended)

python3 -m venv venv
source venv/bin/activate


3. Install dependencies

pip install -r requirements.txt

requirements.txt

yfinance

pandas

numpy

matplotlib

scipy


4. Run the tracker

python portfolio_tracker.py

Optional: specify a custom date range:
python portfolio_tracker.py --start 2018-01-01 --end 2025-08-01

Configuration"

Default tickers and weights are set in the script:

TICKERS = {"ETH": "ETH-USD", "GOLD": "GLD", "NVDA": "NVDA"}
INIT_WEIGHTS = {"ETH": 0.33, "GOLD": 0.34, "NVDA": 0.33}

Change these to track your own portfolio.


Example Console Output

=== Performance Summary ===
Asset Ann. Return Ann. Vol Sharpe Max DD
ETH 48.97% 71.02% 0.65 -93.96%
GOLD 9.27% 12.22% 0.51 -22.00%
NVDA 51.46% 43.12% 1.12 -66.34%
PORTFOLIO 36.20% 40.15% 0.82 -55.00%

=== Daily Return Correlations ===
ETH GOLD NVDA
ETH 1.00 0.10 0.24
GOLD 0.10 1.00 0.06
NVDA 0.24 0.06 1.00

=== Monte Carlo (5y, geometric BM) ===
Assumed annualized μ: 35.00%, σ: 50.00%
5th pct: $55,000 | Median: $390,000 | 95th pct: $1,200,000


Next Steps:

Add rebalancing logic (monthly, quarterly)

Add support for custom holdings (units, cost basis)

Build a Streamlit dashboard for interactive use.


License

MIT License. See LICENSE for details.
