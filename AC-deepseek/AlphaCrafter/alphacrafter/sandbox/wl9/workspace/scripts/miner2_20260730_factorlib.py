"""Shared factor validation utilities for miner_2.

Loads the 15-asset tradable universe (plus observation-only macro series),
truncates data at the current visible date, computes a factor panel, and
produces IC/ICIR/hit-ratio/coverage/turnover/decay metrics at horizon 10
(admission horizon) plus decay horizons 1/2/3/5/20.

Methodology:
- Per-asset close series are used with each asset's own calendar for forward
  returns: fwd_ret_h(t) = close[t+h]/close[t] - 1 on that asset's own rows.
- Daily Spearman rank IC between factor values and forward returns, computed
  on dates with >= 8 valid instruments.
- ICIR = mean(IC)/std(IC).  Hit ratio = fraction of days with IC > 0
  (or IC < 0 if expected direction is negative).
- Turnover: mean absolute rank change per 10 days (rank turnover).
- Coverage: fraction of asset-days with valid factor value.
- Gates (shared benchmark): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADABLE = ["000300.SH", "000688.SH", "SPX", "NDX", "SOX", "HSI", "N225", "SX5E",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
VISIBLE_THROUGH = "2026-07-29"  # current date 2026-07-30 -> previous completed day
MIN_ASSETS = 8

ASSET_ALIAS = {"000300.SH": "000300.SH", "000688.SH": "000688.SH"}


def load_close(symbol: str, macro: bool = False) -> pd.Series:
    d = INDEX_DIR if macro else DATA_DIR
    df = pd.read_csv(os.path.join(d, f"{symbol}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)]
    s = df.set_index("date")["close"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def load_panel() -> pd.DataFrame:
    return pd.concat({a: load_close(a) for a in TRADABLE}, axis=1)


def load_macro() -> dict[str, pd.Series]:
    return {m: load_close(m, macro=True) for m in MACRO}


def factor_panel(panel: pd.DataFrame, fn) -> pd.DataFrame:
    """Apply fn(col_series)->col_series to each asset column; returns aligned panel."""
    out = {}
    for a in panel.columns:
        out[a] = fn(panel[a].dropna())
    return pd.DataFrame(out).sort_index()


def fwd_ret_panel(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(out).sort_index()


def ic_series(fvals: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    idx = fvals.index.intersection(fwd.index)
    ics = {}
    for t in idx:
        x = fvals.loc[t]
        y = fwd.loc[t]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            rho, _ = spearmanr(x[m], y[m])
            if np.isfinite(rho):
                ics[t] = rho
    return pd.Series(ics)


def rank_turnover(fvals: pd.DataFrame, step: int = 10) -> float:
    r = fvals.rank(axis=1)
    dif = r.diff(step).abs()
    return float(dif.mean().mean())


def validate(fvals: pd.DataFrame, fwd10: pd.DataFrame, label: str = "",
             expected_dir: int = 1) -> dict:
    ic = ic_series(fvals, fwd10)
    n = len(ic)
    mean_ic = float(ic.mean()) if n else np.nan
    std_ic = float(ic.std(ddof=1)) if n > 1 else np.nan
    icir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and std_ic > 0 else np.nan
    hit = float((ic * expected_dir > 0).mean()) if n else np.nan
    cov = float(fvals.notna().sum().sum() / (fvals.shape[0] * fvals.shape[1]))
    turn = rank_turnover(fvals)
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        ih = ic_series(fvals, fwd_ret_panel(panel_from(fvals), h))
        decay[str(h)] = round(float(ih.mean()), 4) if len(ih) else np.nan
    return {
        "label": label,
        "n_ic_dates": int(n),
        "ic": round(mean_ic, 4) if np.isfinite(mean_ic) else None,
        "icir": round(icir, 4) if np.isfinite(icir) else None,
        "ic_hit_ratio": round(hit, 4) if np.isfinite(hit) else None,
        "coverage": round(cov, 4),
        "turnover_10d_rank": round(turn, 3),
        "decay_ic_by_horizon": decay,
        "passes": bool(np.isfinite(mean_ic) and np.isfinite(icir)
                       and abs(mean_ic) >= 0.0070 and abs(icir) >= 0.0840),
    }


_panel_cache: pd.DataFrame | None = None


def panel_from(fvals: pd.DataFrame) -> pd.DataFrame:
    global _panel_cache
    if _panel_cache is None:
        _panel_cache = load_panel()
    return _panel_cache.reindex(fvals.index)


if __name__ == "__main__":
    P = load_panel()
    print("panel shape:", P.shape, "rows:", P.index.min().date(), "->", P.index.max().date())
    print("assets with full history:", (P.notna().sum() >= 1500).sum(), "/", P.shape[1])
