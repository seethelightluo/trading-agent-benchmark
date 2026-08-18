"""Shared helpers for miner_3 research cycle on 2026-11-05."""
import json, math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
INDEX_DATA_DIR = "../persistent/index_data/"

IC_THRESHOLD = 0.0070
ICIR_THRESHOLD = 0.0840
MIN_IC_DATES = 60
MIN_ASSETS_PER_DATE = 8


def load_macro(name, days=1200):
    df = pd.read_csv(f"{INDEX_DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).reset_index(drop=True)
    return df


def load_asset(symbol, days=1200):
    df = get_index_daily_data(symbol=symbol, days=days)
    if df is None:
        df = get_stock_daily_data(symbol=symbol, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_close_panel(days=1200):
    closes, dates = {}, None
    vols = {}
    for s in WATCHLIST:
        df = load_asset(s, days=days)
        if df is None or len(df) < 100:
            continue
        closes[s] = df.set_index("date")["close"]
        if "volume" in df.columns:
            vols[s] = df.set_index("date")["volume"]
        dts = df["date"].dt.normalize()
        if dates is None:
            dates = dts
    panel = pd.DataFrame(closes)
    vpanel = pd.DataFrame(vols)
    return panel, vpanel


def forward_returns(panel, horizon=5):
    """Forward return over `horizon` trading days (shift -horizon / 1 - 1)."""
    fwd = panel.shift(-horizon) / panel - 1.0
    return fwd


def spearman_ic(factor_series, fwd_returns, min_assets=MIN_ASSETS_PER_DATE):
    """Cross-sectional Spearman IC per date. factor_series indexed by date, fwd_returns same."""
    ics, dates = [], []
    common = factor_series.index.intersection(fwd_returns.index)
    fs, fr = factor_series.loc[common], fwd_returns.loc[common]
    for dt in common:
        x = fs.loc[dt].dropna()
        y = fr.loc[dt].reindex(x.index)
        m = x.notna() & y.notna()
        if m.sum() < min_assets:
            continue
        xx, yy = x[m], y[m]
        if xx.nunique() < 3 or yy.nunique() < 3:
            continue
        rho = xx.rank().corr(yy.rank())
        if not np.isnan(rho):
            ics.append(rho)
            dates.append(dt)
    return pd.Series(ics, index=dates)


def ic_metrics(ics):
    if len(ics) < MIN_IC_DATES:
        return {"ic": float("nan"), "icir": float("nan"), "n_ic_dates": len(ics),
                "hit": float("nan"), "tstat": float("nan")}
    ic = float(ics.mean())
    sd = float(ics.std(ddof=1))
    icir = ic / sd if sd > 0 else float("nan")
    tstat = ic / (sd / math.sqrt(len(ics))) if sd > 0 else float("nan")
    hit = float((ics > 0).mean())
    return {"ic": ic, "icir": icir, "n_ic_dates": len(ics), "hit": hit, "tstat": tstat}


def coverage(series, panel):
    """Fraction of asset-days with valid factor value (over panel index window)."""
    sub = series[series.index.isin(panel.index)] if hasattr(series.index, "isin") else series
    valid = sub.notna().sum().sum() if hasattr(sub, "sum") else float("nan")
    total = panel.shape[0] * panel.shape[1]
    return valid / total if total else float("nan")


def turnover(series, panel):
    """Mean cross-sectional rank autocorrelation (1 - |d(rank)|/2) -> signal stability 0..1."""
    sub = series[series.index.isin(panel.index)]
    ranks = sub.rank(axis=1)
    diff = ranks.diff().abs().mean(axis=1)
    return float(diff.mean()) if len(diff) else float("nan")


def summarize(name, ics, series, panel, fwd):
    m = ic_metrics(ics)
    m["coverage"] = coverage(series, panel)
    m["turnover_rank_chg"] = turnover(series, panel)
    print(f"[{name}] IC={m['ic']:.4f} ICIR={m['icir']:.4f} n={m['n_ic_dates']} "
          f"hit={m['hit']:.3f} t={m['tstat']:.2f} cov={m['coverage']:.3f} "
          f"turn_rank_chg={m['turnover_rank_chg']:.3f}")
    return m