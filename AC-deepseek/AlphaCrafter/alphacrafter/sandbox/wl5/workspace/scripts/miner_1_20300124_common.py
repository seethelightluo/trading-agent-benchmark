"""Shared helpers for miner_1 factor validation cycle (2030-01-24).

Data rules:
- Use tradable universe of 15 assets from ../persistent/stock_data.
- Macro observation-only signals from ../persistent/index_data.
- Never look past visible_through=2030-01-23 (previous completed trading day).
- Online start 2026-07-16; warm-up 2020-01-01..2026-07-15 is research-only.
"""
import json
import os
import numpy as np
import pandas as pd

VISIBLE_THROUGH = "2030-01-23"
CURRENT_DATE = "2030-01-24"

TRADABLE = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
            "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]

MACRO = ["DXY", "VIX", "USDJPY", "EURUSD", "USDCNY"]

STOCK_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840
CORR_THRESHOLD = 0.50

WARMUP_END = "2026-07-15"


def load_asset(sym):
    df = pd.read_csv(os.path.join(STOCK_DIR, f"{sym}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE_THROUGH].sort_values("date").reset_index(drop=True)
    return df


def load_macro(sym):
    df = pd.read_csv(os.path.join(INDEX_DIR, f"{sym}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE_THROUGH].sort_values("date").reset_index(drop=True)
    return df


def ohlcv_panels(symbols=None):
    """Return dict of DataFrames (dates x assets) for open/high/low/close/volume."""
    symbols = symbols or TRADABLE
    out = {k: {} for k in ["open", "high", "low", "close", "volume"]}
    for s in symbols:
        df = load_asset(s)
        for k in out:
            out[k][s] = pd.Series(df[k].values, index=pd.to_datetime(df["date"].values))
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items()}


def price_panel(field="close", symbols=None):
    return ohlcv_panels(symbols)[field]


def macro_panel(sym):
    df = load_macro(sym)
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"].values))


def fwd_returns(close, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1.0
    return out


def rank_ic_series(factor_panel, fwd_ret, min_valid=8):
    """Daily cross-sectional Spearman IC series (index=date)."""
    idx = factor_panel.index.intersection(fwd_ret.index)
    ics = []
    for d in idx:
        f = factor_panel.loc[d]
        r = fwd_ret.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= min_valid:
            ic = f[m].rank().corr(r[m].rank())
            if not np.isnan(ic):
                ics.append((d, ic))
    if not ics:
        return None
    return pd.Series([x[1] for x in ics], index=[x[0] for x in ics], dtype=float)


def summarize_ic(s, horizon, label=""):
    if s is None or len(s) < 20:
        return None
    ic = s.mean()
    icir = s.mean() / s.std() if s.std() > 0 else 0.0
    return {"label": label, "horizon": horizon, "ic": round(ic, 4),
            "icir": round(icir, 4), "ic_hit_ratio": round((s > 0).mean(), 3),
            "n_ic_dates": int(len(s))}


def coverage_turnover(factor_panel, every=10, min_valid=8):
    valid = factor_panel.notna()
    cov_asset_days = float(valid.sum().sum()) / float(valid.size)
    ge8 = valid.sum(axis=1) >= min_valid
    cov_dates_ge8 = float(ge8.mean())
    to = []
    idx = factor_panel.index[::every]
    for i in range(1, len(idx)):
        d0, d1 = idx[i - 1], idx[i]
        r0 = factor_panel.loc[d0].rank()
        r1 = factor_panel.loc[d1].rank()
        m = r0.notna() & r1.notna()
        if m.sum() >= min_valid:
            to.append((r1[m] - r0[m]).abs().mean())
    turnover = float(np.mean(to)) if to else float("nan")
    return {"coverage_asset_days": round(cov_asset_days, 3),
            "coverage_dates_ge8": round(cov_dates_ge8, 3),
            "turnover_10d_rank": round(turnover, 3)}


def regime_breakdown(ic_series, label=""):
    """IC breakdown by year/period for drift monitoring."""
    if ic_series is None or len(ic_series) < 20:
        return None
    out = {}
    idx = ic_series.index
    for lo, hi in [("2020-01-01", "2026-12-31"), ("2027-01-01", "2027-12-31"),
                   ("2028-01-01", "2028-12-31"), ("2029-01-01", "2029-12-31"),
                   ("2030-01-01", "2035-12-31")]:
        m = (idx >= lo) & (idx <= hi)
        sub = ic_series[m]
        if len(sub) >= 20:
            out[lo[:4]] = {"ic": round(float(sub.mean()), 4),
                           "icir": round(float(sub.mean() / sub.std()), 4),
                           "n": int(len(sub))}
    return out


def encode_signal_artifact(factor_panel):
    """Encode factor signal panel as base64:zlib:csv for provenance/audit."""
    import base64
    import zlib
    df = factor_panel.copy()
    df.index = df.index.strftime("%Y-%m-%d")
    csv = df.to_csv()
    blob = base64.b64encode(zlib.compress(csv.encode("utf-8"))).decode("ascii")
    return {
        "format": "base64:zlib:csv",
        "description": "Factor signal panel: rows = dates, cols = assets. Shape %s" % (list(df.shape),),
        "columns": list(df.columns),
        "shape": list(df.shape),
        "n_valid_values": int(df.notna().sum().sum()),
        "sha256": str(abs(hash(csv)) % 10**16),
        "data": blob,
    }
