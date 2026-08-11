"""Fast vectorized factor validation helpers for miner_3 screening.

Spearman rank IC computed with per-row (cross-sectional) rank + Pearson on
ranks, vectorized across dates (no per-date Python loop). Summary metrics
without decay (decay computed separately only for finalists).
"""
import numpy as np
import pandas as pd


def fast_ic_series(factor, fwd_ret, min_valid=8):
    """Vectorized daily cross-sectional Spearman rank IC.

    factor / fwd_ret: DataFrames dates x assets.
    Returns Series of IC per date.
    """
    factor = factor.reindex(fwd_ret.index)
    valid = factor.notna() & fwd_ret.notna()
    n = valid.sum(axis=1)
    f_rank = factor.rank(axis=1, pct=False).where(valid)
    r_rank = fwd_ret.rank(axis=1, pct=False).where(valid)
    mf = f_rank.sum(axis=1) / n
    mr = r_rank.sum(axis=1) / n
    cf = f_rank.sub(mf, axis=0).where(valid)
    cr = r_rank.sub(mr, axis=0).where(valid)
    cov = (cf * cr).sum(axis=1) / n
    sf = (cf ** 2).sum(axis=1)
    sr = (cr ** 2).sum(axis=1)
    denom = np.sqrt(sf * sr)
    ic = cov / denom.replace(0, np.nan)
    ic = ic.where(n >= min_valid)
    ic = ic.replace([np.inf, -np.inf], np.nan)
    return ic


def fast_summary(ic, factor, close, step=10):
    """ic: Series of daily IC. Returns dict of core admission metrics."""
    ic = ic.dropna()
    n = len(ic)
    if n < 30:
        return None
    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=1))
    icir = float(ic_mean / ic_std) if ic_std and np.isfinite(ic_std) and ic_std > 0 else float("nan")
    hit = float((ic > 0).mean()) if ic_mean >= 0 else float((ic < 0).mean())
    valid_mask = factor.notna()
    coverage_asset_days = float(valid_mask.sum().sum() / (factor.shape[0] * factor.shape[1])) if factor.shape[0] else 0.0
    ge8 = factor.dropna(thresh=8)
    coverage_dates_ge8 = float(len(ge8) / len(factor)) if len(factor) else 0.0
    r = factor.rank(axis=1, pct=True)
    r_step = r.shift(step)
    turn = float((r - r_step).abs().mean().mean()) if r_step.notna().any().any() else float("nan")
    return {
        "ic": round(ic_mean, 4),
        "icir": round(icir, 4) if np.isfinite(icir) else None,
        "ic_hit_ratio": round(hit, 3),
        "n_ic_dates": int(n),
        "coverage_asset_days": round(coverage_asset_days, 3),
        "coverage_dates_ge8": round(coverage_dates_ge8, 3),
        "turnover_10d_rank": round(turn, 3) if np.isfinite(turn) else None,
    }
