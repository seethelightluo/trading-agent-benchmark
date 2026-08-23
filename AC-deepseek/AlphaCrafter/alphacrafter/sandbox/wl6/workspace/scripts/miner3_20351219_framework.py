"""Shared validation framework for factor mining (miner_3).
Uses data only through visible_through date (2035-12-18). 15-instrument
cross-asset universe. Computes cross-sectional IC vs forward returns.
"""
import numpy as np
import pandas as pd
import json, sys

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent"

def load_series(name, idx=False):
    p = os.path.join(DATA_DIR, ("index_data" if idx else "stock_data"), name + ".csv")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

def load_close_panel():
    px = {}
    for a in WATCHLIST:
        # map display names to actual file names
        fname = a
        if a == "SPXX": fname = "SPX"
        if a == "HSG": fname = "HSI"
        if a == "SOXX": fname = "SOX"
        df = load_series(fname)
        px[a] = df["close"]
    panel = pd.DataFrame(px)
    return panel

def clip_to(panel, end="2035-12-18"):
    return panel[panel.index <= end]

def forward_returns(panel, horizon):
    # return over next `horizon` trading days
    fwd = panel.shift(-horizon) / panel - 1.0
    return fwd

def compute_ic_stats(factor_df, fwd_df, horizon):
    """Cross-sectional Spearman IC between factor and forward return per date."""
    ics = []
    dates_used = 0
    assets_ok = 0
    for dt in fwd_df.index:
        f = factor_df.loc[dt] if dt in factor_df.index else pd.Series(dtype=float)
        r = fwd_df.loc[dt]
        # align by asset
        m = pd.concat([f, r], axis=1).dropna()
        if len(m) < 8:
            continue
        if m.iloc[:,0].nunique() < 3 or m.iloc[:,1].nunique() < 3:
            continue
        import scipy.stats as st
        ic, _ = st.spearmanr(m.iloc[:,0], m.iloc[:,1])
        if np.isnan(ic):
            continue
        dates.append(ic)
        d_used = dt
        assets_ok += len(m)
    if not dates:
        return None
    ics = np.array(dates)
    ic = float(np.mean(ics))
    icir = float(np.mean(ics)/ (np.std(ics)+1e-12)) if len(ics)>1 else 0.0
    hit = float(np.mean(ics > 0))
    return {
        "ic": ic, "icir": icir, "ic_hit_ratio": hit,
        "n_ic_dates": len(ics), "avg_assets_per_date": assets_ok/len(ics),
        "mean_abs_ic": float(np.mean(np.abs(ics))),
    }

def decay_profile(factor_df, panel, horizons=(1,2,3,5,10,20)):
    out = {}
    for h in horizons:
        fwd = forward_returns_panel(panel, h)
        r = compute_ic(factor_df, fwd, h)
        if r:
            out[h] = round(r["ic"],4)
    return out

if __name__ == "__main__":
    panel = load_factor_panel()
    panel = validate_to(panel)
    print("panel shape", panel.shape, "last date", panel.index.max())