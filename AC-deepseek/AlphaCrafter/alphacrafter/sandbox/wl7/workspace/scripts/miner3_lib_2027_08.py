"""miner_3 shared validation library (2027-08-31 cycle).

Universe: 15 tradable cross-asset instruments. Data capped at MAX_VISIBLE
(2027-08-30 = last completed trading day before current date 2027-08-31).
IC = cross-sectional Spearman rank IC per date (>= 8 valid assets).
Admission gates (benchmark contract): |IC| >= 0.007 and |ICIR| >= 0.084 @ h=10.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_VISIBLE = "2027-08-30"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084}
HORIZONS = (1, 2, 3, 5, 10, 20)
EPS = 1e-12

_CACHE: dict = {}


def load_panel() -> pd.DataFrame:
    if "panel" in _CACHE:
        return _CACHE["panel"]
    closes = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        closes[s] = df["close"].astype(float)
    panel = pd.concat(closes, axis=1, sort=True)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    _CACHE["panel"] = panel
    return panel


def load_ohlc_volume() -> dict[str, pd.DataFrame]:
    if "ohlcv" in _CACHE:
        return _CACHE["ohlcv"]
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[s] = df[["open", "close", "high", "low", "volume"]].astype(float)
    _CACHE["ohlcv"] = out
    return out


def load_macro() -> dict[str, pd.Series]:
    if "macro" in _CACHE:
        return _CACHE["macro"]
    out = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        out[m] = df["close"].astype(float)
    _CACHE["macro"] = out
    return out


def fwd_returns(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward h-observation return on each asset's own calendar, reindexed."""
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(cols, index=panel.index)


def rank_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC per date."""
    ics, idxs = [], []
    idx = factor.index.intersection(fwd.index)
    for d in idx:
        f = factor.loc[d]
        r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < MIN_ASSETS:
            continue
        ics.append(f[m].rank().corr(r[m].rank()))
        idxs.append(d)
    return pd.Series(ics, index=pd.DatetimeIndex(idxs))


def summarize_ic(ic: pd.Series, label: str = "") -> dict:
    if len(ic) == 0:
        return {"label": label, "n": 0, "ic": np.nan, "icir": np.nan,
                "hit": np.nan, "ic_std": np.nan}
    ic = ic.dropna()
    mu, sd = ic.mean(), ic.std(ddof=1)
    return {
        "label": label,
        "n": int(len(ic)),
        "ic": float(mu),
        "icir": float(mu / sd * np.sqrt(len(ic))) if sd > EPS else 0.0,
        "hit": float((ic > 0).mean()),
        "ic_std": float(sd),
    }


def validate_factor(factor_df: pd.DataFrame, panel: pd.DataFrame,
                    h: int = 10, direction: float = 1.0,
                    per_year: bool = True) -> dict:
    fwd = fwd_returns(panel, h)
    ic = rank_ic_series(factor_df * direction, fwd)
    out = summarize_ic(ic, f"h{h}")
    if per_year:
        years = {}
        for y, g in ic.groupby(ic.index.year):
            years[str(y)] = summarize_ic(g, str(y))
        out["per_year"] = years
    return out


def decay_profile(factor_df: pd.DataFrame, panel: pd.DataFrame,
                  direction: float = 1.0) -> dict:
    out = {}
    for h in HORIZONS:
        fwd = fwd_returns(panel, h)
        ic = rank_ic_series(factor_df * direction, fwd)
        s = summarize_ic(ic, f"h{h}")
        out[str(h)] = s["ic"]
    return out


def turnover_10d(factor_df: pd.DataFrame, panel: pd.DataFrame) -> float:
    """Mean cross-sectional rank change over 10d windows, averaged."""
    ranks = factor_df.rank(axis=1)
    diffs = []
    for i in range(10, len(ranks)):
        a = ranks.iloc[i - 10]
        b = ranks.iloc[i]
        m = a.notna() & b.notna()
        if m.sum() >= MIN_ASSETS:
            diffs.append((a[m] - b[m]).abs().mean())
    return float(np.mean(diffs)) if diffs else np.nan


def coverage(factor_df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    valid = factor_df.notna()
    return {
        "coverage_asset_days": float(valid.mean().mean()),
        "coverage_dates_ge8": float((valid.sum(axis=1) >= MIN_ASSETS).mean()),
    }
