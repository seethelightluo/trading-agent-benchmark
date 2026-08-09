"""Fast vectorized IC analysis for miner3 factor screening."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close

MIN_NAMES = 8


def fwd_returns(closes, fwd_days=1):
    cols = {}
    for s, df in closes.items():
        cols[s] = df["close"].shift(-fwd_days) / df["close"] - 1.0
    return pd.DataFrame(cols)


def fast_ic(factor_df, fwd_df, min_names=MIN_NAMES, rank=True):
    """Vectorized daily cross-sectional IC between factor and forward return."""
    idx = factor_df.index.intersection(fwd_df.index)
    F = factor_df.reindex(idx).astype(float)
    R = fwd_df.reindex(idx).astype(float)
    if rank:
        F = F.rank(axis=1)
        R = R.rank(axis=1)
    Fv = F.values
    Rv = R.values
    mask = np.isfinite(Fv) & np.isfinite(Rv)
    n = mask.sum(axis=1)
    ok = n >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan,
                "hit": np.nan, "ic_std": np.nan}
    Fm = np.where(mask, Fv, 0.0)
    Rm = np.where(mask, Rv, 0.0)
    sx = Fm.sum(axis=1)
    sy = Rm.sum(axis=1)
    sxx = (Fm * Fm).sum(axis=1)
    syy = (Rm * Rm).sum(axis=1)
    sxy = (Fm * Rm).sum(axis=1)
    with np.errstate(all="ignore"):
        num = n * sxy - sx * sy
        den = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
        ic = num / den
    ic = ic[ok]
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan,
                "hit": np.nan, "ic_std": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(n[ok].sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean()), "ic_std": float(ic.std())}


def fast_ic_all(factor_df, closes, horizons=(1, 2, 3, 5, 10, 20, 30), min_names=MIN_NAMES):
    out = {}
    for h in horizons:
        fwd = fwd_returns(closes, h)
        out[h] = fast_ic(factor_df, fwd, min_names)
    return out


def turnover10(factor_df, rebal=10):
    ranks = factor_df.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna()
        cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        chg.append((cur[common] - prev[common]).abs().mean() / (len(common) - 1))
    return float(np.mean(chg)) if chg else np.nan


def coverage(factor_df, closes):
    n_total = 0
    n_valid = 0
    for s in closes:
        if s in factor_df.columns:
            n_valid += len(factor_df[s].dropna())
        n_total += len(closes[s])
    return n_valid / n_total if n_total else np.nan


def make_panel(closes, fn):
    cols = {}
    for s, df in closes.items():
        try:
            fv = fn(df)
            if fv is not None and len(fv):
                cols[s] = fv
        except Exception as e:
            print(f"  [warn] {s}: {e}")
    return pd.DataFrame(cols)