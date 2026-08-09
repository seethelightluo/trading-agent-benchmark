"""Miner2 shared fast vectorized helpers: IC analysis on common-date panels."""
import numpy as np
import pandas as pd

MIN_NAMES = 8


def fwd_returns(closes, fwd_days=1):
    cols = {}
    for s, df in closes.items():
        cols[s] = df["close"].shift(-fwd_days) / df["close"] - 1.0
    return pd.DataFrame(cols)


def fast_ic(factor_df, fwd_df, min_names=MIN_NAMES, rank=True):
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


def ic_all(factor_df, closes, horizons=(1, 2, 3, 5, 10, 20, 30), min_names=MIN_NAMES):
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


def coverage_panel(factor_df, n_total_cells):
    return float(factor_df.notna().sum().sum()) / n_total_cells if n_total_cells else np.nan


def screen(name, panel, fwd1, fwd5, fwd10, n_cells, verbose=True):
    cov = coverage_panel(panel, n_cells)
    to = turnover10(panel)
    ic1 = fast_ic(panel, fwd1)
    ic5 = fast_ic(panel, fwd5)
    ic10 = fast_ic(panel, fwd10)
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:28s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": cov, "to": to, "ic1": ic1, "ic5": ic5,
            "ic10": ic10, "passed": passed}
