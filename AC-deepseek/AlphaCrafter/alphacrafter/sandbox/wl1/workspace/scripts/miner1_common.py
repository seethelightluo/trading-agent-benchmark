"""Shared utilities for miner_1 factor exploration & validation.
Universe: 15 tradable cross-asset instruments + 5 observation-only macro series.
"""
import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"


def load_closes(end_date="2029-01-18", start_date="2020-01-01", min_rows=300):
    """Load close prices for tradable universe, aligned on common trading dates."""
    frames = {}
    for sym in TRADABLE:
        df = pd.read_csv(f"{DATA_DIR}/{sym}.csv", parse_dates=["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.set_index("date").sort_index()
        frames[sym] = df["close"].rename(sym)
    px = pd.concat(frames, axis=1)
    px = px.dropna(how="all")
    # require most symbols present on each date
    px = px.dropna(thresh=max(8, int(len(TRADABLE) * 0.6)))
    px = px.ffill().dropna(how="all")
    return px


def load_macro(end_date="2029-01-18", start_date="2020-01-01"):
    frames = {}
    for sym in MACRO:
        df = pd.read_csv(f"{INDEX_DIR}/{sym}.csv", parse_dates=["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.set_index("date").sort_index()
        frames[sym] = df["close"].rename(sym)
    mx = pd.concat(frames, axis=1)
    return mx


def forward_returns(px, horizons=(1, 5, 10)):
    """Forward returns per horizon (in fractional terms)."""
    out = {}
    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        out[f"fwd{h}"] = fwd
    return out


def daily_ic(factor_df, fwd_ret):
    """Cross-sectional Spearman IC per date (require >=8 valid instruments)."""
    ic = []
    dates = []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() >= 8:
            ic.append(f[mask].rank().corr(r[mask].rank()))
            dates.append(dt)
    s = pd.Series(ic, index=pd.DatetimeIndex(dates))
    return s


def summarize_ic(ic_s, label=""):
    ic = ic_s.dropna()
    if len(ic) == 0:
        return {"label": label, "n_dates": 0, "ic": np.nan, "icir": np.nan,
                "hit": np.nan, "ic_std": np.nan}
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = (ic > 0).mean()
    return {"label": label, "n_dates": len(ic), "ic": mean_ic, "icir": icir,
            "hit": hit, "ic_std": std_ic}


def print_summary(res, gate_ic=0.0070, gate_icir=0.0840):
    for k, v in res.items():
        if isinstance(v, dict) and "ic" in v:
            ic, icir = v["ic"], v["icir"]
            passed = (abs(ic) >= gate_ic) and (abs(icir) >= gate_icir)
            print(f"[{k}] n={v['n_dates']} IC={ic:+.4f} ICIR={icir:+.3f} "
                  f"hit={v['hit']:.3f} {'PASS' if passed else 'fail'}")


def coverage(factor_df):
    """Fraction of tradable instruments with non-NaN on each date, avg."""
    cov = factor_df.notna().mean(axis=1)
    return cov.mean(), cov.min()


def turnover(factor_df):
    """Cross-sectional rank turnover: mean |rank_t - rank_{t-1}| / (N-1)."""
    ranks = factor_df.rank(axis=1)
    chg = ranks.diff().abs().mean(axis=1)
    n = ranks.shape[1]
    return (chg / (n - 1)).mean()
