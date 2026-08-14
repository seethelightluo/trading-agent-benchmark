"""Common data loading + IC validation utilities for miner_1.
Restricts all data to visible_through (2035-01-03) to avoid lookahead.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VISIBLE = "2035-01-03"
WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

DATA_DIR = "../persistent/stock_data"
MACRO_DIR = "../persistent/index_data"


def load_close(symbol, data_dir=DATA_DIR):
    df = pd.read_csv(f"{data_dir}/{symbol}.csv", parse_dates=["date"])
    df = df[df["date"] <= VISIBLE].sort_values("date").reset_index(drop=True)
    return df


def load_all():
    closes = {}
    rets = {}
    for s in WATCHLIST:
        df = load_close(s)
        closes[s] = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
        rets[s] = closes[s].pct_change()
    for s in MACRO:
        df = pd.read_csv(f"{MACRO_DIR}/{s}.csv", parse_dates=["date"])
        df = df[df["date"] <= VISIBLE].sort_values("date").reset_index(drop=True)
        closes[s] = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]))
        rets[s] = closes[s].pct_change()
    close_df = pd.DataFrame(closes).sort_index()
    ret_df = pd.DataFrame(rets).sort_index()
    return close_df, ret_df


def forward_returns(close_df, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        out[h] = close_df.shift(-h) / close_df - 1.0
    return out


def eval_factor(factor_df, fwd, min_valid=8, label="factor"):
    """factor_df: date x asset DataFrame of factor values (already restricted).
    fwd: date x asset DataFrame of forward returns (h days).
    Returns dict of metrics.
    """
    dates = factor_df.index.intersection(fwd.index)
    ics = {}
    for d in dates:
        f = factor_df.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        idx = f.index.intersection(r.index)
        if len(idx) < min_valid:
            continue
        ic, _ = spearmanr(f.loc[idx], r.loc[idx])
        if np.isnan(ic):
            continue
        ics[d] = ic
    ic_series = pd.Series(ics).sort_index()
    if len(ic_series) < 20:
        return {"error": "too few IC dates", "n": len(ic_series)}
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit = (ic_series > 0).mean()
    # recency: last 252 dates
    last = ic_series.tail(252)
    last_ic = last.mean()
    last_icir = last.mean() / last.std() if last.std() > 0 else 0.0
    # coverage
    valid = factor_df.notna()
    coverage = valid.sum().sum() / (factor_df.shape[0] * factor_df.shape[1])
    # turnover: mean abs change of cross-sectional rank per 10d
    ranks = factor_df.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean()
    return {
        "label": label,
        "n_ic_dates": len(ic_series),
        "first_date": str(ic_series.index[0].date()),
        "last_date": str(ic_series.index[-1].date()),
        "ic": float(mean_ic),
        "icir": float(icir),
        "ic_hit_ratio": float(hit),
        "last252_ic": float(last_ic),
        "last252_icir": float(last_icir),
        "coverage": float(coverage),
        "turnover_10d_rank": float(to),
        "ic_std": float(std_ic),
        "min_ic": float(ic_series.min()),
        "max_ic": float(ic_series.max()),
    }


def yearly_ic(factor_df, fwd, min_valid=8):
    dates = factor_df.index.intersection(fwd.index)
    rows = []
    for d in dates:
        f = factor_df.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        idx = f.index.intersection(r.index)
        if len(idx) < min_valid:
            continue
        ic, _ = spearmanr(f.loc[idx], r.loc[idx])
        if not np.isnan(ic):
            rows.append((d, ic))
    s = pd.Series({d: v for d, v in rows}).sort_index()
    out = {}
    for y, g in s.groupby(s.index.year):
        out[y] = (round(float(g.mean()), 4), round(float(g.mean() / g.std()), 3) if g.std() > 0 else 0.0, len(g))
    return out


def print_metrics(m):
    print(f"--- {m['label']} ---")
    for k in ["n_ic_dates", "first_date", "last_date", "ic", "icir", "ic_hit_ratio",
              "last252_ic", "last252_icir", "coverage", "turnover_10d_rank", "ic_std"]:
        print(f"  {k}: {m[k]}")
    print(f"  gate: |IC|>={0.007} {'PASS' if abs(m['ic']) >= 0.007 else 'FAIL'} | "
          f"|ICIR|>={0.084} {'PASS' if abs(m['icir']) >= 0.084 else 'FAIL'}")
