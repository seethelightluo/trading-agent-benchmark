"""
miner3_20270408_exp01_sharpe_mom.py
Explore Rolling Sharpe ratio (risk-adjusted momentum) as a factor.
Idea: momentum normalized by volatility should provide more consistent cross-asset signals
across equity indices, commodities, crypto, and yields.
Uses 63-day (~3 month) lookback for both return and volatility estimation.
"""

import json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
from scipy.stats import spearmanr

WATCHLIST = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E",
    "000688.SH", "SOX", "NDX", "XAU", "COPPER",
    "WTI", "BTC", "ETH", "US10Y", "CN10Y"
]

CURRENT_DATE = "2027-04-08"
LOOKBACK = 63
MIN_DAYS = LOOKBACK + 20
ADMISSION_HORIZON = 10


def compute_rolling_sharpe(close: pd.Series, lookback=63) -> pd.Series:
    """Compute rolling Sharpe ratio = mean return / std return"""
    ret = close.pct_change()
    min_periods = max(lookback // 2, 30)
    mean_ret = ret.rolling(lookback, min_periods=min_periods).mean()
    std_ret = ret.rolling(lookback, min_periods=min_periods).std()
    sharpe = mean_ret / std_ret.replace(0, np.nan)
    return sharpe


def compute_ic_one_horizon(factor_df, ret_df, horizon=10):
    """
    Cross-sectional Spearman IC for each date.
    factor_df: DataFrame indexed by date, columns = symbols
    ret_df: forward return DataFrame (same shape)
    Returns: list of (date, ic, n_valid)
    """
    results = []
    common_dates = sorted(set(factor_df.index) & set(ret_df.index))
    for d in common_dates:
        f = factor_df.loc[d].dropna()
        r = ret_df.loc[d].dropna()
        valid = f.index.intersection(r.index)
        if len(valid) >= 8:
            fv = f[valid].values
            rv = r[valid].values
            # Handle constant or near-constant factor values
            if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
                continue
            corr, _ = spearmanr(fv, rv)
            if not np.isnan(corr):
                results.append((d, corr, len(valid)))
    return results


def main():
    print(f"=== Sharpe Ratio (rolling {LOOKBACK}d) Factor Exploration ===")
    print(f"Current date: {CURRENT_DATE}")
    print(f"Admission horizon: {ADMISSION_HORIZON}d forward\n")

    # Load data
    days_needed = LOOKBACK + 2 * ADMISSION_HORIZON + 50
    print(f"Loading {days_needed} days of data for {len(WATCHLIST)} instruments...")
    data = {}
    for sym in WATCHLIST:
        df = get_stock_daily_data(symbol=sym, days=days_needed)
        if df is not None and len(df) >= MIN_DAYS:
            data[sym] = df
        else:
            print(f"  SKIP {sym}: {len(df) if df is not None else 0} days")

    print(f"Loaded {len(data)}/{len(WATCHLIST)} instruments successfully\n")

    # Build price matrix
    price_dict = {}
    for sym, df in data.items():
        price_dict[sym] = df.set_index('date')['close']
    prices = pd.DataFrame(price_dict)
    prices = prices.sort_index()

    print(f"Price matrix shape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Number of dates: {len(prices)}\n")

    # Compute factor: rolling Sharpe ratio
    sharpe_df = prices.apply(lambda col: compute_rolling_sharpe(col, LOOKBACK))
    # Shift factor by 1 to avoid lookahead
    factor = sharpe_df.shift(1)
    print(f"Factor computed, shape: {factor.shape}")

    # Compute forward returns at different horizons
    horizons = [1, 2, 3, 5, 10, 20]
    ic_results = {}

    for h in horizons:
        forward_ret = prices.pct_change(h).shift(-h)
        ic_obs = compute_ic_one_horizon(factor, forward_ret, h)
        ics = [x[1] for x in ic_obs]
        n_vals = [x[2] for x in ic_obs]

        if len(ics) > 0:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / std_ic if std_ic > 0 else 0
            hit_ratio = np.mean([1 for ic in ics if ic > 0])
            print(f"Horizon {h:2d}d: IC={mean_ic:.6f}, ICIR={icir:.6f}, "
                  f"Hit={hit_ratio:.3f}, N_dates={len(ics)}, "
                  f"Avg_N_assets={np.mean(n_vals):.1f}")
            ic_results[h] = {
                'ic': mean_ic,
                'icir': icir,
                'hit_ratio': hit_ratio,
                'n_dates': len(ics),
                'avg_n_assets': np.mean(n_vals)
            }
        else:
            print(f"Horizon {h:2d}d: No valid IC observations")
            ic_results[h] = None

    # Print admission check for horizon=10
    h = ADMISSION_HORIZON
    r = ic_results.get(h)
    if r:
        ic10 = r['ic']
        icir10 = r['icir']
        print(f"\n--- Admission Gate Check (horizon={h}d) ---")
        print(f"IC = {ic10:.6f}  (threshold: |IC| >= 0.0070)")
        print(f"ICIR = {icir10:.6f}  (threshold: |ICIR| >= 0.0840)")
        passes_ic = abs(ic10) >= 0.0070
        passes_icir = abs(icir10) >= 0.0840
        if passes_ic and passes_icir:
            print(">>> PASSES admission gate -- candidate for persistence")
        else:
            print(">>> FAILS admission gate")
            if not passes_ic:
                print(f"    IC {ic10:.6f} < 0.0070 threshold")
            if not passes_icir:
                print(f"    ICIR {icir10:.6f} < 0.0840 threshold")
    else:
        print("\nNo 10-day horizon IC data available")

    # Coverage analysis
    valid_dates = factor.notna().sum(axis=1)
    pct_full = (valid_dates >= len(WATCHLIST)).mean()
    pct_ge8 = (valid_dates >= 8).mean()
    print(f"\nCoverage: % full coverage={pct_full:.3f}, % >=8 assets={pct_ge8:.3f}")

    # Turnover: rank correlation stability
    factor_ranks = factor.rank(axis=1, method='average')
    turnover_10d = []
    for i in range(10, len(factor_ranks)):
        prev = factor_ranks.iloc[i - 10]
        curr = factor_ranks.iloc[i]
        valid = prev.dropna().index.intersection(curr.dropna().index)
        if len(valid) >= 8:
            from scipy.stats import spearmanr as sp
            rho, _ = sp(prev[valid], curr[valid])
            turnover_10d.append(1 - rho if not np.isnan(rho) else 0)
    avg_turnover = np.mean(turnover_10d) if turnover_10d else 0
    print(f"Avg 10d rank turnover: {avg_turnover:.4f}")

    print("\n=== Exploration Complete ===")


if __name__ == "__main__":
    main()