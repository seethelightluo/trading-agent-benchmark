"""Cross-sectional factor validation harness.
Matches benchmark admission contract: compute daily cross-sectional IC of a
factor panel vs forward 10-day returns (skip t+1), across the warm-up window
2020-01-01..2026-07-15 (and optionally a recent window).
"""
import numpy as np
import pandas as pd
import os, json

WARM_START = "2020-01-02"
WARM_END = "2026-07-15"
ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225",
          "NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"

def load_close():
    closes = {}
    for a in ASSETS:
        df = pd.read_csv(f"{DATA}/{a}.csv")
        df = df[["date","close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        closes[a] = df.set_index("date")["close"]
    panel = pd.DataFrame(closes)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    return panel

def load_index(name):
    df = pd.read_csv(f"{IDX}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date","close"]].set_index("date")["close"]
    return df[~df.index.duplicated(keep="last")].sort_index()

def fwd_ret_10(panel, h=10, skip=1):
    """Forward h-day return, skipping `skip` days ahead."""
    fr = panel.shift(-h) / panel.shift(-skip) - 1.0
    return fr

def cross_sectional_ic(factor_panel, fwd_panel, min_cov=8):
    """Daily cross-sectional Pearson IC between factor and forward return.
    aligned by date, requires >= min_cov valid assets per date."""
    common = factor_panel.index.intersection(fwd_panel.index)
    ics, ndates, covs = [], 0, []
    all_dates = fwd_panel.index
    n_assets = fwd_panel.shape[1]
    for d in common:
        f = factor_panel.loc[d]
        r = fwd_panel.loc[d]
        m = f.isna() | r.isna()
        fv = f[~m]; rv = r[~m]
        if len(fv) < min_cov:
            continue
        if fv.std() < 1e-12 or rv.std() < 1e-12:
            continue
        with np.errstate(all="ignore"):
            ic = np.corrcoef(fv.values, rv.values)[0, 1]
        if np.isnan(ic):
            continue
        ics.append(ic)
        ndates += 1
        covs.append(len(fv))
    ics = np.array(ics)
    res = {
        "ic": float(np.mean(ics)),
        "icir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
        "ic_hit_ratio": float((ics > 0).mean()),
        "n_ic_dates": ndates,
        "mean_cov": float(np.mean(covs)) if covs else np.nan,
        "coverage_dates_ge8": ndates / len(all_dates),
    }
    return res

def window(df, start=WARM_START, end=WARM_END):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    return df.loc[(df.index >= s) & (df.index <= e)]

def run(factor_panel, fwd10, label="", min_cov=8, extra_windows=None):
    print(f"\n===== {label} =====")
    ws = {"warmup(2020..2026-07)": (WARM_START, WARM_END)}
    if extra_windows:
        ws.update(extra_windows)
    for nm, (s, e) in ws.items():
        fp = window(factor_panel, s, e)
        fr = window(fwd10, s, e)
        r = cross_sectional_ic(fp, fr, min_cov)
        print(f"{nm:32s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} hit={r['ic_hit_ratio']:.2f} "
              f"ndates={r['n_ic_dates']} cov={r['mean_cov']:.1f}")
    return r