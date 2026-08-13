"""Shared validation harness for miner_2 (cross-asset 15-instrument universe).

Data source: ../persistent/stock_data/*.csv and ../persistent/index_data/*.csv
Visible cutoff: 2033-09-14 (visible through current simulation date 2033-09-15).
No lookahead: factor at t uses data up to t; forward return uses t+1..t+h.
Each asset has its own trading calendar; factor and forward-return computations are
done per-asset on the asset's OWN calendar, then reindexed to the union panel index
for cross-sectional IC.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
VISIBLE_THROUGH = "2033-09-14"

DATA_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")


def load_asset(symbol: str) -> pd.DataFrame:
    p = (INDEX_DIR if symbol in MACRO else DATA_DIR) / f"{symbol}.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _panel(column: str) -> pd.DataFrame:
    frames = {}
    for a in TRADABLES:
        df = load_asset(a)
        frames[a] = pd.Series(df[column].astype(float).values,
                              index=pd.to_datetime(df["date"]), name=a)
    return pd.concat(frames, axis=1).sort_index()


def load_close_panel() -> pd.DataFrame:
    return _panel("close")


def load_volume_panel() -> pd.DataFrame:
    return _panel("volume")


def load_ohlc_panels():
    """Return dict of panels for open/high/low/close (union index)."""
    return {k: _panel(k) for k in ["open", "high", "low", "close"]}


def macro_series(name: str) -> pd.Series:
    df = load_asset(name)
    return pd.Series(df["close"].astype(float).values,
                     index=pd.to_datetime(df["date"]), name=name)


def per_asset(panel: pd.DataFrame, func, *args, **kwargs) -> pd.DataFrame:
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = func(s, *args, **kwargs).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def fwd_ret_series(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-h) / s - 1.0


def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return per_asset(panel, fwd_ret_series, horizon)


def compute_ic(factor_panel: pd.DataFrame, ret_panel: pd.DataFrame,
               min_assets: int = 8) -> pd.Series:
    dates = factor_panel.index.intersection(ret_panel.index)
    F = factor_panel.loc[dates]
    R = ret_panel.loc[dates]
    Fr = F.rank(axis=1).values
    Rr = R.rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= min_assets
    ics = np.full(len(dates), np.nan)
    idx = np.where(valid)[0]
    for i in idx:
        f = Fr[i, m[i]]
        r = Rr[i, m[i]]
        f = f - f.mean()
        r = r - r.mean()
        denom = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / denom if denom > 0 else np.nan
    return pd.Series(ics, index=dates, name="ic")


def panel_rank_corr(a: pd.DataFrame, b: pd.DataFrame, min_assets: int = 8) -> float:
    dates = a.index.intersection(b.index)
    Ar = a.loc[dates].rank(axis=1).values
    Br = b.loc[dates].rank(axis=1).values
    m = (~np.isnan(Ar)) & (~np.isnan(Br))
    valid = m.sum(axis=1) >= min_assets
    cs = []
    idx = np.where(valid)[0]
    for i in idx:
        x = Ar[i, m[i]]
        y = Br[i, m[i]]
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        if denom > 0:
            cs.append((x * y).sum() / denom)
    return float(np.mean(cs)) if cs else float("nan")


def summarize_ic(ic: pd.Series, label: str = "", min_obs: int = 30) -> dict:
    s = ic.dropna()
    out = {"label": label, "n_ic_dates": int(len(s))}
    if len(s) >= min_obs:
        out["ic_mean"] = round(float(s.mean()), 4)
        out["ic_std"] = round(float(s.std(ddof=1)), 4)
        out["icir"] = round(float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))), 3) if s.std(ddof=1) > 0 else None
        out["ic_hit"] = round(float((s > 0).mean()), 3)
    else:
        out["ic_mean"] = None
        out["icir"] = None
    return out


def regime_ic(ic: pd.Series, bounds=None, labels=None) -> dict:
    if bounds is None:
        bounds = [(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
                  (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
                  (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")),
                  (pd.Timestamp("2027-01-01"), pd.Timestamp("2030-12-31")),
                  (pd.Timestamp("2031-01-01"), pd.Timestamp("2033-09-14"))]
        labels = ["2020-2021", "2022-2024", "2025-2026", "2027-2030", "2031-2033"]
    out = {}
    for (lo, hi), lab in zip(bounds, labels):
        sub = ic[(ic.index >= lo) & (ic.index <= hi)].dropna()
        if len(sub) >= 30:
            out[lab] = {"ic": round(float(sub.mean()), 4),
                        "icir": round(float(sub.mean() / sub.std(ddof=1) * np.sqrt(len(sub))), 3)
                        if sub.std(ddof=1) > 0 else None,
                        "n": int(len(sub))}
    last = ic.dropna().tail(250)
    if len(last) >= 30:
        out["last250"] = {"ic": round(float(last.mean()), 4),
                          "icir": round(float(last.mean() / last.std(ddof=1) * np.sqrt(len(last))), 3)
                          if last.std(ddof=1) > 0 else None,
                          "n": int(len(last))}
    return out


def turnover_10d(factor_panel: pd.DataFrame) -> float:
    """Mean 10-day rank-change turnover (0..1 scale)."""
    Fr = factor_panel.rank(axis=1)
    chg = Fr.diff(10).abs()
    n = Fr.notna().sum(axis=1)
    valid = n >= 8
    if valid.sum() == 0:
        return float("nan")
    sub = chg[valid]
    scale = sub.max(axis=1).clip(lower=1)  # max possible rank change
    return float((sub / scale).mean().mean())


def coverage_metrics(factor_panel: pd.DataFrame) -> dict:
    n_assets = factor_panel.shape[1]
    valid = factor_panel.notna()
    cov_days = float(valid.mean().mean())
    dates_ge8 = (valid.sum(axis=1) >= 8).mean()
    return {"coverage_asset_days": round(cov_days, 4),
            "coverage_dates_ge8": round(float(dates_ge8), 4),
            "n_assets": n_assets,
            "n_dates": int(len(factor_panel))}


def decay_ic(factor_panel: pd.DataFrame, close_panel: pd.DataFrame,
             horizons=(1, 2, 3, 5, 10, 20)) -> dict:
    out = {}
    for h in horizons:
        ret = forward_returns(close_panel, h)
        ic = compute_ic(factor_panel, ret)
        s = ic.dropna()
        out[str(h)] = round(float(s.mean()), 4) if len(s) >= 30 else None
    return out
