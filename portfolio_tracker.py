#!/usr/bin/env python3
"""
Portfolio Tracker (server-friendly, updated)
- Fixes yfinance 'Adj Close' / MultiIndex issues (uses adjusted Close; group_by='column')
- Forces clean single-level column names (ETH, GOLD, NVDA)
- Aligns weights safely to columns
- Saves charts to PNG (no GUI needed)
- Handles short histories (rolling corr window auto-adjusts)
"""

import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf

# Use non-GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- DEFAULT CONFIG --------
TICKERS = {
    "ETH": "ETH-USD",
    "GOLD": "GLD",     # or "XAUUSD" if you prefer spot and it loads well
    "NVDA": "NVDA",
}
START = "2018-01-01"
END = None
RISK_FREE_ANNUAL = 0.03
INIT_WEIGHTS = {
    "ETH": 0.33,
    "GOLD": 0.34,
    "NVDA": 0.33
}
OUTDIR = "outputs"  # where to save charts

# -------- HELPERS --------
def annualize_return(daily_ret: pd.Series) -> float:
    return (1 + daily_ret.mean())**252 - 1

def annualize_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std() * np.sqrt(252)

def sharpe_ratio(daily_ret: pd.Series, rf=RISK_FREE_ANNUAL) -> float:
    ann_ret = annualize_return(daily_ret)
    ann_vol = annualize_vol(daily_ret)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol

def max_drawdown(cum_curve: pd.Series) -> float:
    roll_max = cum_curve.cummax()
    dd = cum_curve / roll_max - 1.0
    return dd.min()

def download_prices(ticker_map, start=START, end=END) -> pd.DataFrame:
    """
    Robust downloader:
    - yfinance auto_adjust=True (Close is adjusted)
    - group_by='column' to avoid MultiIndex
    - renames each series to alias (ETH, GOLD, NVDA)
    """
    frames = []
    for alias, ticker in ticker_map.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,    # adjusted Close by default
            group_by="column"    # prevent MultiIndex
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
        if col is None:
            raise ValueError(f"No usable price column for {ticker}. Columns: {list(df.columns)}")

        s = df[[col]].copy()
        s.columns = [alias]  # force clean name
        frames.append(s)

    prices = pd.concat(frames, axis=1)
    prices = prices.sort_index().ffill().dropna(how="any")  # drop rows missing any asset
    # Safety: flatten if any accidental MultiIndex sneaks in
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [c[-1] if isinstance(c, tuple) else c for c in prices.columns]
    return prices

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def portfolio_series(prices: pd.DataFrame, weights: dict):
    # Align weights strictly to DataFrame columns (missing → 0)
    w = pd.Series(weights, index=prices.columns).fillna(0.0)
    rets = daily_returns(prices)
    port_ret = (rets * w).sum(axis=1)
    cum = (1 + port_ret).cumprod()
    return port_ret, cum, rets

def summary_table(asset_rets: pd.DataFrame, port_ret: pd.Series, port_cum: pd.Series) -> pd.DataFrame:
    rows = []
    for col in asset_rets.columns:
        r = asset_rets[col]
        rows.append([
            col,
            f"{annualize_return(r):.2%}",
            f"{annualize_vol(r):.2%}",
            f"{sharpe_ratio(r):.2f}",
            f"{max_drawdown((1+r).cumprod()):.2%}"
        ])
    rows.append([
        "PORTFOLIO",
        f"{annualize_return(port_ret):.2%}",
        f"{annualize_vol(port_ret):.2%}",
        f"{sharpe_ratio(port_ret):.2f}",
        f"{max_drawdown(port_cum):.2%}"
    ])
    return pd.DataFrame(rows, columns=["Asset", "Ann. Return", "Ann. Vol", "Sharpe", "Max DD"])

def rolling_corr(asset_rets: pd.DataFrame, a="ETH", b="NVDA", window=90) -> pd.Series:
    # Auto-adjust window if history is short
    if len(asset_rets) < 5:
        return pd.Series(dtype=float)
    win = min(window, max(2, len(asset_rets) // 5))  # ensure at least small window
    return asset_rets[a].rolling(win).corr(asset_rets[b]).dropna()

def monte_carlo_terminal_values(port_ret: pd.Series, n_years=5, n_sims=5000, init_value=100_000):
    mu = port_ret.mean() * 252
    sigma = port_ret.std() * np.sqrt(252)
    T = n_years
    z = np.random.normal(size=n_sims)
    terminal = init_value * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    return terminal, mu, sigma

def save_chart(fig, filename):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")

# -------- MAIN --------
def main():
    parser = argparse.ArgumentParser(description="Portfolio tracker (ETH, GOLD, NVDA). Saves charts to PNG.")
    parser.add_argument("--start", default=START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=END, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    prices = download_prices(TICKERS, args.start, args.end)
    port_ret, port_cum, rets = portfolio_series(prices, INIT_WEIGHTS)

    # Summary & correlations (simple, single-level columns)
    summary = summary_table(rets, port_ret, port_cum)
    corr = rets.corr()

    print("\n=== Performance Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Daily Return Correlations ===")
    print(corr.round(2).to_string())

    # Charts
    # 1) Normalized prices
    fig1, ax1 = plt.subplots(figsize=(10,5))
    (prices / prices.iloc[0]).plot(ax=ax1, title="Normalized Prices (Indexed to 1.0)")
    save_chart(fig1, "normalized_prices.png")

    # 2) Portfolio cumulative growth
    fig2, ax2 = plt.subplots(figsize=(10,5))
    port_cum.plot(ax=ax2, title="Portfolio Cumulative Growth")
    save_chart(fig2, "portfolio_cumulative_growth.png")

    # 3) Rolling correlation ETH vs NVDA (skip gracefully if empty)
    rc = rolling_corr(rets, a="ETH", b="NVDA", window=90)
    if rc.empty:
        print("\n[info] Rolling correlation series empty (history too short). Skipping plot.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10,4))
        rc.plot(ax=ax3, title="Rolling Correlation (ETH vs NVDA)")
        save_chart(fig3, "rolling_corr_ETH_NVDA.png")

    # Monte Carlo (5 years)
    if len(port_ret) > 0:
        terminal, mu, sigma = monte_carlo_terminal_values(port_ret, n_years=5, n_sims=10_000, init_value=100_000)
        p5, p50, p95 = np.percentile(terminal, [5, 50, 95])

        print("\n=== Monte Carlo (5y, geometric BM) ===")
        print(f"Assumed annualized μ: {mu:.2%}, σ: {sigma:.2%}")
        print(f"5th pct: ${p5:,.0f} | Median: ${p50:,.0f} | 95th pct: ${p95:,.0f}")
    else:
        print("\n[info] Not enough portfolio return history for Monte Carlo.")

if __name__ == "__main__":
    main()

