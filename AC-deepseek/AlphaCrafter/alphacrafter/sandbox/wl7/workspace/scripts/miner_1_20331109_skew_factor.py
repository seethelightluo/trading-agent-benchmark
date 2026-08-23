"""miner_1 explore: return-distribution skewness factor.

Idea: assets exhibiting positive skew in recent daily returns (fat right tail,
frequent small up-moves plus occasional large rallies) vs negative skew can
carry different forward risk/reward across the cross-asset universe. This is
orthogonal to kurtosis (tail weight) and to downside-vol-ratio, and may offer
diversification to the momentum-driven ensemble.

Construction: rolling skewness of daily returns over window W, using a skip to
avoid microstructure/lagged correlation bleed (like other library factors).

Gates (15-asset benchmark): abs(daily IC) >= 0.0070, abs(ICIR) >= 0.0840.
Validation period: full sample 2020-01-01 .. visible_through (2033-11-08).
Per-year breakdown to assess regime robustness; recent drift monitoring.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_shared import (ASSETS, load_close, load_macro, forward_ret,
                          daily_ic, ic_stats, summarize, coverage_stats,
                          IC_GATE, ICIR_GATE, MIN_VALID_PER_DATE)

VISIBLE = "2033-11-08"


def lib_skew(close, window=20, skip=5, min_periods=12):
    """CS demeaned rolling skewness of daily returns (skip lag bias)."""
    ret = close.pct_change()
    r = ret.shift(skip)
    sk = r.rolling(window, min_periods=min_periods).skew()
    return sk.subtract(sk.median(axis=1), axis=0)


def lib_abs_skew(close, window=20, skip=5, min_periods=12):
    ret = close.pct_change()
    r = ret.shift(skip)
    sk = r.rolling(window, min_periods=min_periods).skew()
    return sk.abs().subtract(sk.abs().median(axis=1), axis=0)


def lib_voladj_skew(close, window=20, skip=5, min_periods=12):
    ret = close.pct_change()
    r = ret.shift(skip)
    sk = r.rolling(window, min_periods=min_periods).skew()
    vol = r.rolling(window, min_periods=min_periods).std()
    adj = (sk.abs() / vol).replace([np.inf, -np.inf], np.nan)
    return adj.subtract(adj.median(axis=1), axis=0)


def report(name, factor, close, years=True):
    fwd_h10 = forward_ret(close, 10)
    ic10 = daily_ic(factor, fwd_h10)
    st10 = ic_stats(ic10, 10)
    cov = coverage_stats(factor, fwd_h10)
    to = np.nan
    try:
        from miner_shared import rank_turnover
        to = rank_turnover(factor)
    except Exception:
        pass
    print(f"=== {name} ===")
    print(f"  h10 IC={st10['ic']:.5f} ICIR={st10['icir']:.5f} hit={st10['hit']:.3f} n={st10['n']}")
    print(f"  coverage_asset_days={cov['coverage_asset_days']:.3f} dates_ge8={cov['coverage_dates_ge8']:.3f} turn={to:.3f}")
    decay = summarize(factor, close)
    print("  decay IC by h:", {str(k): round(v['ic'], 4) for k, v in decay.items()})
    if years:
        icf = ic10.to_frame("ic")
        icf["year"] = icf.index.year
        for yr, grp in icf.groupby("year"):
            s = grp["ic"].dropna()
            if len(s) == 0:
                continue
            m, sd = s.mean(), s.std(ddof=1)
            print(f"    {yr}: ic={m:.4f} icir={m/sd if sd>0 else np.nan:.4f} n={len(s)}")
    return st10


def main():
    close = load_close(VISIBLE)
    print(f"universe n_assets={close.shape[1]}  dates={close.shape[0]}  {close.index.min()}..{close.index.max()}")
    print(f"admission gates: IC>={IC_GATE} ICIR>={ICIR_GATE} min_valid_per_date={MIN_VALID_PER_DATE}\n")

    for name, fn in [
        ("skew_20d_skip5", lib_skew),
        ("abs_skew_20d_skip5", lib_abs_skew),
        ("voladj_skew_20d_skip5", lib_voladj_skew),
    ]:
        try:
            fac = fn(close)
        except Exception as e:
            print(f"{name} ERROR {e}")
            continue
        report(name, fac, close)

        # Recent-sample drift (post-warmup / post-ensemble era)
        for lab, start in [("recent_2027", "2027-01-01"), ("recent_2031", "2031-01-01")]:
            sub_close = close.loc[close.index >= start]
            sub_fac = fac.loc[close.index >= start]
            if len(sub_close) < 200:
                continue
            fwd = forward_ret(sub_close, 10)
            ic = daily_ic(sub_fac, fwd)
            s = ic_stats(ic, 10)
            print(f"    [{lab}] ic={s['ic']:.4f} icir={s['icir']:.4f} n={s['n']}")


if __name__ == "__main__":
    main()