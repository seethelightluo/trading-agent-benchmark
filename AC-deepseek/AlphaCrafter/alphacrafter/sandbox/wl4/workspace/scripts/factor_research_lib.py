"""Shared factor research library for miner_1.

Loads the 15-asset tradable universe + macro signals through the previous
completed trading day (no lookahead), computes factor panels, forward returns,
rank-IC metrics, and correlation vs existing library factors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]
LIB_DIR = Path("factors")


def load_panels(days: int = 3000):
    """Return {asset: DataFrame} for tradable + macro, full history to last completed day."""
    panels = {}
    for s in TRADABLE:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is not None and len(df) > 60:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            panels[s] = df
    for s in MACRO:
        df = get_index_daily_data(symbol=s, days=days)
        if df is not None and len(df) > 60:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            panels[s] = df
    return panels


def close_panel(panels, assets=None):
    assets = assets or TRADABLE
    return pd.concat(
        {a: panels[a]["close"].astype(float) for a in assets if a in panels},
        axis=1,
    ).sort_index()


def ret_panel(panels, assets=None):
    c = close_panel(panels, assets)
    return c.pct_change()


def forward_returns(closes, horizon: int):
    """Forward h-day return computed from closes: close[t+h]/close[t]-1."""
    return closes.shift(-horizon) / closes - 1.0


def rank_ic_series(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8):
    """Daily Spearman rank IC between factor cross-section and forward return cross-section."""
    dates, ics = [], []
    for dt in factor_panel.index:
        f = factor_panel.loc[dt]
        r = fwd.loc[dt] if dt in fwd.index else None
        if r is None:
            continue
        pair = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
        if len(pair) < min_valid or pair["r"].std() < 1e-14 or pair["f"].std() < 1e-14:
            continue
        ic = pair["f"].corr(pair["r"], method="spearman")
        if not math.isnan(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")


def summarize_ic(ic_series: pd.Series, expected_sign: int = 1):
    ic = ic_series.mean()
    icir = ic_series.mean() / ic_series.std(ddof=1) if ic_series.std(ddof=1) > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean()) if expected_sign else float((np.sign(ic_series) != 0).mean())
    return {
        "ic": round(float(ic), 4),
        "icir": round(float(icir), 4),
        "ic_hit_ratio": round(float(hit), 3),
        "n_ic_dates": int(len(ic_series)),
        "ic_std": round(float(ic_series.std(ddof=1)), 4),
    }


def coverage_metrics(factor_panel: pd.DataFrame, assets=None, min_valid=8):
    assets = assets or TRADABLE
    valid = factor_panel.notna()
    asset_days = float(valid[list(valid.columns)].sum().sum())
    total_days = float(factor_panel.shape[0] * len([a for a in assets if a in valid.columns]))
    dates_ge8 = float((valid.sum(axis=1) >= min_valid).mean())
    return {
        "coverage_asset_days": round(asset_days / total_days, 3) if total_days else 0.0,
        "coverage_dates_ge8": round(dates_ge8, 3),
    }


def turnover_rank(factor_panel: pd.DataFrame, step: int = 10):
    """Mean absolute raw-rank change over `step` days (0..N-1 scale)."""
    r = factor_panel.rank(axis=1, method="average")
    diff = r.diff(step).abs().mean().mean()
    return round(float(diff), 3) if not math.isnan(diff) else None


def decay_profile(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20), min_valid=8, expected_sign=1):
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, h)
        ics = rank_ic_series(factor_panel, fwd, min_valid)
        if len(ics):
            out[str(h)] = round(float(ics.mean()), 4)
    return out


def library_signals(panels, closes=None, rets=None, vix=None):
    """Recompute existing library factor panels from their stored definitions."""
    closes = closes if closes is not None else close_panel(panels)
    rets = rets if rets is not None else closes.pct_change()
    if vix is None:
        vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
    sig = {}
    # mom_10d_skip5
    sig["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
    # mom_120d_skip5
    sig["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
    # vol_of_vol20x60
    sig["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    # vix_beta_cond_60x20: -beta(asset,VIX,60) * (VIX/VIX.shift(20)-1)
    if vix is not None:
        vix_ret = vix.pct_change()
        beta = {}
        for a in rets.columns:
            z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
            b = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            beta[a] = b
        beta_df = pd.DataFrame(beta, index=rets.index)
        cond = -beta_df * (vix / vix.shift(20) - 1.0)
        sig["vix_beta_cond_60x20"] = cond
    return sig


def max_library_corr(candidate: pd.DataFrame, library: dict):
    """Max absolute pairwise Pearson correlation of stacked cross-sectional values."""
    best, best_key = 0.0, None
    for name, lib_sig in library.items():
        both = pd.concat([candidate.stack().rename("cand"), lib_sig.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, best_key = abs(r), name
    return round(best, 4), best_key


def load_library_meta():
    """Load persisted factor json defs to know expression/direction."""
    meta = {}
    if LIB_DIR.exists():
        for p in sorted(LIB_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                meta[d["factor_id"]] = d
            except Exception:
                pass
    return meta


def full_eval(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20), min_valid=8,
              expected_sign=1, library=None, admission_horizon=10):
    """Compute all validation metrics at the admission horizon plus decay + library corr."""
    fwd = forward_returns(closes, admission_horizon)
    ics = rank_ic_series(factor_panel, fwd, min_valid)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(factor_panel, min_valid=min_valid))
    m["turnover_10d_rank"] = turnover_rank(factor_panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(factor_panel, closes, horizons, min_valid, expected_sign)
    if library is not None:
        corr, key = max_library_corr(factor_panel, library)
        m["max_abs_library_correlation"] = corr
        m["max_corr_factor"] = key
    return m, ics
