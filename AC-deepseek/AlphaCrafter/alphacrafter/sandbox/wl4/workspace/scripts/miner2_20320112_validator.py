"""Shared factor validation framework for miner_2.
Loads 15-asset price panel + macro signals, computes factor values,
forward returns, and IC/ICIR/turnover/coverage metrics across regimes.

Usage: import functions or run standalone for a factor function dict.
"""
import numpy as np
import pandas as pd
import json, glob, hashlib, zlib, base64, os

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
DATA_DIR = "../persistent/stock_data"
MACRO_DIR = "../persistent/index_data"
CUTOFF = "2032-01-12"

def load_panel(cutoff=CUTOFF):
    closes = {}
    for s in WATCH:
        df = pd.read_csv(f"{DATA_DIR}/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= cutoff].set_index("date")["close"].rename(s)
        closes[s] = df
    px = pd.DataFrame(closes).sort_index()
    macro = {}
    for m in MACRO:
        df = pd.read_csv(f"{MACRO_DIR}/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= cutoff].set_index("date")["close"].rename(m)
        macro[m] = df
    mx = pd.DataFrame(macro).sort_index()
    return px, mx

def forward_returns(px, horizons=(1,2,3,5,10,20)):
    fwd = {}
    for h in horizons:
        fwd[h] = px.shift(-h) / px - 1.0
    return fwd

def daily_ic(factor_df, fwd_df, min_valid=8):
    """Cross-sectional Spearman IC per date."""
    dates, ics = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_df.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method="spearman")
        if np.isfinite(ic):
            dates.append(dt)
            ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def full_metrics(factor_df, fwd_map, min_valid=8, horizons=(1,2,3,5,10,20)):
    out = {}
    for h in horizons:
        ic_s = daily_ic(factor_df, fwd_map[h], min_valid)
        if len(ic_s) == 0:
            out[h] = {"ic": np.nan, "icir": np.nan, "hit": np.nan, "n": 0}
            continue
        out[h] = {
            "ic": float(ic_s.mean()),
            "icir": float(ic_s.mean() / ic_s.std()) if ic_s.std() > 0 else 0.0,
            "hit": float((ic_s > 0).mean()),
            "n": int(len(ic_s)),
            "ic_std": float(ic_s.std()),
        }
    # coverage
    valid = factor_df.notna().sum().sum()
    total = factor_df.shape[0] * factor_df.shape[1]
    cov_asset_days = valid / total if total else 0
    dates_ge8 = int((factor_df.notna().sum(axis=1) >= min_valid).sum())
    cov_dates_ge8 = dates_ge8 / factor_df.shape[0] if factor_df.shape[0] else 0
    # turnover: mean abs change in cross-sectional rank, per 10d
    ranks = factor_df.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean()
    return {
        "horizons": {str(h): out[h] for h in horizons},
        "coverage_asset_days": float(cov_asset_days),
        "coverage_dates_ge8": float(cov_dates_ge8),
        "n_dates_total": int(factor_df.shape[0]),
        "turnover_10d_rank": float(to) if np.isfinite(to) else np.nan,
    }

def max_library_corr(factor_df):
    """Max abs pairwise correlation vs existing library signal artifacts (if present)."""
    best = {"corr": 0.0, "factor": None}
    for f in glob.glob("factors/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        sig = d.get("validation", {}).get("signal_artifact", {})
        if not sig:
            continue
        try:
            raw = base64.b64decode(sig["data"])
            arr = pd.read_csv(pd.io.common.BytesIO(zlib.decompress(raw)), index_col=0)
            arr.index = pd.to_datetime(arr.index)
            common = factor_df.index.intersection(arr.index)
            if len(common) < 60:
                continue
            a = factor_df.loc[common].values.astype(float)
            b = arr.loc[common].values.astype(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 500:
                continue
            # pairwise per-asset correlation of aligned values
            corrs = []
            for j in range(a.shape[1]):
                mm = m[:, j]
                if mm.sum() < 30:
                    continue
                c = np.corrcoef(a[mm, j], b[mm, j])[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
            if corrs:
                mc = float(np.max(np.abs(corrs)))
                if mc > best["corr"]:
                    best = {"corr": mc, "factor": d.get("factor_id")}
        except Exception:
            continue
    return best

def make_signal_artifact(factor_df):
    """base64:zlib:csv artifact for provenance + gate recovery."""
    s = factor_df.round(8).to_csv()
    comp = zlib.compress(s.encode())
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {list(factor_df.shape)}",
        "columns": list(factor_df.columns),
        "shape": list(factor_df.shape),
        "n_valid_values": int(factor_df.notna().sum().sum()),
        "sha256": hashlib.sha256(comp).hexdigest()[:16],
        "data": base64.b64encode(comp).decode(),
    }

if __name__ == "__main__":
    print("validator loaded OK")
