"""
Shared metrics engine for miner_1 factor validation.
Universe: 15 tradable cross-asset instruments.
Research window: factor dates 2020-01-01..2026-07-15; data visible through 2026-07-29.
Admission gates: |IC|>=0.007, |ICIR|>=0.084 at horizon 10 (benchmark-wide contract).
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

WATCH = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MAX_VISIBLE = "2026-07-29"
FACTOR_LAST = "2026-07-15"
MIN_ASSETS = 8
ADMISSION = {"ic": 0.007, "icir": 0.084}

def load_panel(assets=None):
    assets = assets or WATCH
    frames = {}
    for s in assets:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        frames[s] = df
    return frames

def load_macro(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    return df

def panel_col(frames, col):
    return pd.DataFrame({s: f[col].astype(float) for s, f in frames.items()}).sort_index()

def evaluate(fvals, closes, horizon=10, min_assets=MIN_ASSETS, factor_last=FACTOR_LAST,
             label=None):
    """fvals: date x asset factor DataFrame. Returns metrics dict or None."""
    fwd = closes.shift(-horizon) / closes - 1.0
    rows = []
    for dt in fvals.index:
        if dt > pd.Timestamp(factor_last):
            continue
        f = fvals.loc[dt]; r = fwd.loc[dt]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n < min_assets:
            continue
        ic, _ = spearmanr(f[m], r[m])
        rows.append((dt, ic, n))
    if len(rows) < 200:
        print(f"{label}: INSUFFICIENT dates ({len(rows)})")
        return None
    ics = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    ic_mean = float(ics.mean()); ic_std = float(ics.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ics > 0).mean())
    ranks = fvals.rank(axis=1)
    turn = float((ranks - ranks.shift(10)).abs().mean(axis=1).mean())
    cov_ad = float(fvals.notna().sum().sum() / (fvals.shape[0] * fvals.shape[1]))
    cov_dt = float((fvals.notna().sum(axis=1) >= min_assets).mean())
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwdh = closes.shift(-h) / closes - 1.0
        hs = []
        for dt in fvals.index:
            if dt > pd.Timestamp(factor_last):
                continue
            f = fvals.loc[dt]; r = fwdh.loc[dt]
            m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
            if m.sum() < min_assets: continue
            ic, _ = spearmanr(f[m], r[m])
            hs.append(ic)
        decay[str(h)] = round(float(np.mean(hs)), 4) if hs else None
    return {
        "ic": round(ic_mean, 4), "icir": round(icir, 4),
        "ic_hit_ratio": round(hit, 3), "n_ic_dates": len(rows),
        "ic_std": round(ic_std, 4), "turnover_10d_rank": round(turn, 3),
        "coverage_asset_days": round(cov_ad, 3), "coverage_dates_ge8": round(cov_dt, 3),
        "decay_ic_by_horizon": decay,
    }

def gate_pass(res):
    return res is not None and abs(res["ic"]) >= ADMISSION["ic"] and abs(res["icir"]) >= ADMISSION["icir"]

def library_corr(fvals, closes, library_ids=None, n_days=500):
    """Max abs mean per-date cross-sectional rank corr with library factors.
    Library factors recomputed from signal definitions. Returns (max_abs, per_factor)."""
    libs = {}
    for fid in (library_ids or []):
        f = library_signal(fid, closes)
        if f is not None:
            libs[fid] = f
    out = {}
    common = fvals.index.intersection(closes.index)
    for fid, lf in libs.items():
        cs = []
        for dt in common[-n_days:]:
            f = fvals.loc[dt]; g = lf.reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f) & np.isfinite(g)
            if m.sum() >= MIN_ASSETS:
                r, _ = spearmanr(f[m], g[m])
                cs.append(r)
        out[fid] = round(float(np.mean(cs)), 4) if cs else None
    valid = [abs(v) for v in out.values() if v is not None]
    return (round(max(valid), 4) if valid else None), out

def library_signal(fid, closes):
    """Recompute library factor signals from close prices (definitions from factors/)."""
    rets = closes.pct_change()
    if fid == "mom_10d_skip5":
        return closes.shift(5) / closes.shift(15) - 1.0
    if fid == "mom_120d_skip5":
        return closes.shift(5) / closes.shift(125) - 1.0
    if fid == "vol_of_vol20x60":
        v = rets.rolling(20).std()
        return v.rolling(60).std()
    if fid == "vix_beta_cond_60x20":
        try:
            vix = load_macro("VIX")["close"].astype(float)
            vixr = vix.pct_change()
            beta = rets.rolling(60).cov(vixr) / vixr.rolling(60).var()
            return -beta * (vix / vix.shift(20) - 1.0)
        except Exception:
            return None
    return None
