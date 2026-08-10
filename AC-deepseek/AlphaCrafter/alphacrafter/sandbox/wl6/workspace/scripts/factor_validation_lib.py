"""Shared factor validation framework for the 15-instrument cross-asset universe.

Usage:
    from factor_validation_lib import load_panel, load_macro, ic_analysis, rank_ic_series

Pipeline mirrors the persisted library metrics:
  - Factor values computed on daily bars through visible_through (previous completed day).
  - Forward return horizon H: ret_{t -> t+H} using close prices.
  - Daily rank IC: Spearman corr between cross-sectional factor ranks and fwd return ranks,
    on dates with >= 8 valid instruments (small-universe rule).
  - ICIR = mean(IC)/std(IC); hit ratio = fraction of dates with IC>0 (for positive direction).
  - Turnover: mean absolute rank change per 10 trading days.
  - Coverage: fraction of (asset, day) cells with valid factor value; fraction of dates with >=8 valid.
  - Decay: IC at horizons 1,2,3,5,10,20.
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
            "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_INSTR = 8
INDEX_DATA = Path("../persistent/index_data")


def load_panel(max_date: str | None = None) -> pd.DataFrame:
    """Close-price panel (assets x dates) via the simulator API, cut at max_date."""
    closes = {}
    for sym in TRADABLE:
        try:
            df = get_stock_daily_data(symbol=sym, days=4000)
        except Exception:
            df = None
        if df is not None and "close" in df and len(df) > 30:
            s = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
            closes[sym] = s
    panel = pd.DataFrame(closes).sort_index()
    if max_date is not None:
        panel = panel[panel.index <= pd.Timestamp(max_date)]
    return panel


def load_macro(name: str, max_date: str | None = None) -> pd.Series:
    path = INDEX_DATA / f"{name}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["close"].astype(float)
    if max_date is not None:
        s = s[s.index <= pd.Timestamp(max_date)]
    return s


def align_fwd_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward close-to-close returns over `horizon` trading days (asset x date)."""
    fwd = panel.shift(-horizon) / panel - 1.0
    return fwd


def rank_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily Spearman rank IC between factor values and forward returns."""
    ics = {}
    dates = factor.index.intersection(fwd.index)
    for d in dates:
        f = factor.loc[d].dropna()
        r = fwd.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < MIN_INSTR:
            continue
        fc, rc = f[common].values, r[common].values
        if np.std(fc) == 0 or np.std(rc) == 0:
            continue
        rho, _ = __spearman(fc, rc)
        ics[d] = rho
    return pd.Series(ics, dtype=float)


def __spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import rankdata, spearmanr
    return spearmanr(x, y)


def ic_analysis(factor: pd.DataFrame, panel: pd.DataFrame, horizon: int = 10,
                label: str = "") -> dict:
    fwd = align_fwd_returns(panel, horizon)
    ic = rank_ic_series(factor, fwd)
    ic_abs_mean = float(ic.mean())
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")
    icir = ic_mean / ic_std if ic_std and math.isfinite(ic_std) and ic_std > 0 else float("nan")
    hit = float((ic > 0).mean()) if len(ic) else float("nan")
    # coverage
    valid = factor.notna()
    cov_asset_days = float(valid.mean().mean()) if len(valid) else 0.0
    cov_dates = float((valid.sum(axis=1) >= MIN_INSTR).mean()) if len(valid) else 0.0
    # turnover: mean abs rank change per 10 trading days
    rank = factor.rank(axis=1, pct=True)
    dr = rank.diff().abs()
    turn = float(dr.mean(axis=1).mean() * 10) if len(dr) else float("nan")
    # decay
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        ic_h = rank_ic_series(factor, align_fwd_returns(panel, h))
        decay[str(h)] = round(float(ic_h.mean()), 4) if len(ic_h) else None
    out = {
        "label": label,
        "horizon": horizon,
        "n_ic_dates": int(len(ic)),
        "ic": round(ic_abs_mean, 4) if math.isfinite(ic_abs_mean) else None,
        "ic_signed": round(ic_mean, 4) if math.isfinite(ic_mean) else None,
        "icir": round(icir, 4) if math.isfinite(icir) else None,
        "ic_hit_ratio": round(hit, 3) if math.isfinite(hit) else None,
        "coverage_asset_days": round(cov_asset_days, 3),
        "coverage_dates_ge8": round(cov_dates, 3),
        "turnover_10d_rank": round(turn, 3) if math.isfinite(turn) else None,
        "decay_ic_by_horizon": decay,
        "ic_std": round(ic_std, 4) if math.isfinite(ic_std) else None,
    }
    return out


def print_report(res: dict):
    print(f"--- {res['label']} (horizon={res['horizon']}) ---")
    for k, v in res.items():
        if k in ("decay_ic_by_horizon",):
            print(f"  {k}: {v}")
        elif k != "label":
            print(f"  {k}: {v}")


def library_corr(factor: pd.DataFrame, library: dict[str, pd.DataFrame]) -> float:
    """Max absolute pairwise rank correlation of factor vs persisted library signal panels."""
    f = factor.rank(axis=1, pct=True)
    best = 0.0
    for fid, sig in library.items():
        s = sig.reindex(f.index).rank(axis=1, pct=True)
        ics = []
        for d in f.index.intersection(s.index):
            a, b = f.loc[d], s.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= MIN_INSTR:
                rho, _ = __spearman(a[m].values, b[m].values)
                ics.append(rho)
        if ics:
            best = max(best, abs(float(np.mean(ics))))
    return best
