"""
miner2_20341026_validate_lib.py
Shared validation library (self-contained) for miner_2 factor mining.
Uses persistent CSV data directly (no simulator calls). Validation window
ends 2034-10-25 (latest completed trading day). No lookahead: factor uses
data through t; forward return t -> t+h.

Universe: 15 tradable cross-asset instruments. Admission gate:
|IC| >= 0.0070 and |ICIR| >= 0.0840 at 10-day horizon.
"""
import json
import base64
import zlib
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
CURRENT_DATE = pd.Timestamp("2034-10-25")
IC_GATE = 0.0070
ICIR_GATE = 0.0840
MIN_ASSETS_PER_DATE = 8
MIN_N_IC_DATES = 120


def load_all(end_date=CURRENT_DATE):
    closes, vols, opens, highs, lows = {}, {}, {}, {}, {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        closes[a] = df["close"].astype(float)
        if "volume" in df.columns:
            vols[a] = df["volume"].astype(float)
        if "open" in df.columns:
            opens[a] = df["open"].astype(float)
        if "high" in df.columns:
            highs[a] = df["high"].astype(float)
        if "low" in df.columns:
            lows[a] = df["low"].astype(float)
    close = pd.DataFrame(closes)
    vol = pd.DataFrame(vols) if vols else None
    open_ = pd.DataFrame(opens) if opens else None
    high = pd.DataFrame(highs) if highs else None
    low = pd.DataFrame(lows) if lows else None
    # macro observation-only signals
    macro = {}
    for name in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]:
        df = pd.read_csv(f"{INDEX_DIR}/{name}.csv", parse_dates=["date"])
        df = df[df["date"] <= end_date].set_index("date").sort_index()
        macro[name] = df["close"].astype(float)
    macro = pd.DataFrame(macro)
    return close, vol, open_, high, low, macro


def union_panel(close):
    """Union of dense per-asset calendars reindexed to common index."""
    idx = close.index
    dense = {}
    for a in ASSETS:
        c = close[a].dropna()
        dense[a] = c.reindex(idx)
    return pd.DataFrame(dense)


def factor_panel(fn, close, vol, open_, high, low, macro, **params):
    """Apply factor fn per asset on its dense calendar, reindexed to union panel."""
    vals = {}
    for a in ASSETS:
        c = close[a].dropna().sort_index()
        v = vol[a].dropna().sort_index() if vol is not None else None
        o = open_[a].dropna().sort_index() if open_ is not None else None
        h = high[a].dropna().sort_index() if high is not None else None
        lo = low[a].dropna().sort_index() if low is not None else None
        m = macro.reindex(c.index)
        try:
            s = fn(a, c, v, o, h, lo, m, **params)
            vals[a] = pd.Series(np.asarray(s, dtype=float), index=c.index).reindex(close.index)
        except Exception as e:
            vals[a] = pd.Series(np.nan, index=close.index)
    return pd.DataFrame(vals)


def fwd_returns(close, horizon):
    """Per-asset forward returns on union panel index."""
    out = {}
    for a in ASSETS:
        c = close[a].dropna().sort_index()
        fr = (c.shift(-horizon) / c - 1.0).reindex(close.index)
        out[a] = fr
    return pd.DataFrame(out)


def ic_series(factor, fwd_ret, min_assets=MIN_ASSETS_PER_DATE):
    dates, ics = [], []
    for dt in factor.index:
        x = factor.loc[dt]
        y = fwd_ret.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            xv = x[m].astype(float).values
            yv = y[m].astype(float).values
            if np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                rho = spearmanr(xv, yv)[0]
                if np.isfinite(rho):
                    ics.append(rho)
                    dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def turnover_rank(factor, lag=10):
    ranks = factor.rank(axis=1)
    d = ranks.diff(lag).abs().mean(axis=1)
    return float(d.dropna().mean())


def coverage(factor):
    n_total = float(factor.notna().sum().sum())
    denom = factor.shape[0] * factor.shape[1]
    ge8 = float((factor.notna().sum(axis=1) >= MIN_ASSETS_PER_DATE).mean())
    return n_total / denom, ge8


def validate_factor(fn, close, vol, open_, high, low, macro,
                    horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10,
                    min_assets=MIN_ASSETS_PER_DATE, **params):
    panel = factor_panel(fn, close, vol, open_, high, low, macro, **params)
    cov_ad, cov_ge8 = coverage(panel)
    decay, ic_by_h = {}, {}
    for h in horizons:
        fr = fwd_returns(close, h)
        ic = ic_series(panel, fr, min_assets=min_assets)
        ic_by_h[h] = ic
        decay[h] = float(ic.mean()) if len(ic) else np.nan
    ic_main = ic_by_h[admission_horizon]
    ic = float(ic_main.mean()) if len(ic_main) else np.nan
    icir = float(ic_main.mean() / ic_main.std()) if len(ic_main) > 2 else np.nan
    hit = float((ic_main > 0).mean()) if len(ic_main) else np.nan
    if np.isfinite(ic) and ic < 0:
        hit = float((ic_main < 0).mean())
    return {
        "panel": panel,
        "ic": ic,
        "icir": icir,
        "ic_hit_ratio": hit,
        "n_ic_dates": int(len(ic_main)),
        "coverage_asset_days": round(cov_ad, 4),
        "coverage_dates_ge8": round(cov_ge8, 4),
        "turnover_10d_rank": round(turnover_rank(panel), 4),
        "decay_ic_by_horizon": {str(h): round(decay[h], 4) for h in horizons},
    }


def artifact_b64(panel):
    csv_text = panel.to_csv()
    compressed = zlib.compress(csv_text.encode())
    return base64.b64encode(compressed).decode()


def library_panels():
    """Load existing effective factor panels from factors/ JSON signal artifacts."""
    lib = {}
    import os
    for f in sorted(os.listdir("factors")):
        if not f.endswith(".json") or "reason" in f or f == "factor_ensemble.json":
            continue
        try:
            d = json.load(open(f"factors/{f}"))
            art = d.get("validation", {}).get("signal_artifact", {}).get("data")
            if not art:
                continue
            raw = base64.b64decode(art)
            csv_tex