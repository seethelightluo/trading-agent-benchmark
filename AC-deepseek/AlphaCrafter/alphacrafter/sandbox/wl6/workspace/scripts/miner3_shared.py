"""Shared validation framework for factor mining (miner_3).
Uses data only through visible_through date (2035-12-18). 15-instrument
cross-asset universe. Computes cross-sectional Spearman IC vs forward returns.
"""
import numpy as np
import pandas as pd
import os

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent"
END_DATE = "2035-12-18"

def load_series(name, idx=False):
    p = os.path.join(DATA_DIR, ("index_data" if idx else "stock_data"), name + ".csv")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

def load_close_panel(end=END_DATE):
    px = {}
    for a in WATCH:
        df = load_series(a)
        px[a] = df["close"]
    panel = pd.DataFrame(px)
    panel = panel[panel.index <= pd.Timestamp(end)]
    return panel

def load_macro(name, end=END_DATE):
    df = load_series(name, idx=True)
    return df[df.index <= pd.Timestamp(end)]

def forward_return_panel(panel, horizon):
    return panel.shift(-horizon) / panel - 1.0

def compute_ic(factor_df, fwd_df):
    import scipy.stats as st
    ics = []
    assets_ok = 0
    for dt in fwd_df.index:
        if dt not in factor_df.index:
            continue
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        x = pd.concat([f, r], axis=1).dropna()
        if len(x) < 8:
            continue
        if x.iloc[:,0].nunique() < 3 or x.iloc[:,1].nunique() < 3:
            continue
        ic, _ = st.spearmanr(x.iloc[:,0], x.iloc[:,1])
        if np.isnan(ic):
            continue
        ics.append(ic)
        assets_ok += len(x)
    if not ics:
        return None
    ics = np.array(ics)
    return {
        "ic": float(np.mean(ics)),
        "icir": float(np.mean(ics)/ (np.std(ics)+1e-12)),
        "ic_hit_ratio": float(np.mean(ics > 0)),
        "mean_abs_ic": float(np.mean(np.abs(ics))),
        "n_ic_dates": len(ics),
        "avg_assets_per_date": assets_ok/len(ics),
    }

def decay_profile(factor_df, panel, horizons=(1,2,3,5,10,20)):
    out = {}
    for h in horizons:
        fwd = forward_return_panel(panel, h)
        r = compute_ic(factor_df, fwd)
        out[h] = round(r["ic"],4) if r else None
    return out