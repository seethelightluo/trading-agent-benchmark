"""miner_2 2026-07-30 -- calendar seasonality factor screen.

Idea: per-asset historical average return in the current calendar month,
computed from PRIOR years only (rolling 3-year same-month window, shifted back
1 year to avoid lookahead). Assets with strong historical seasonal patterns
(e.g. commodities demand seasons, equity January/December effects, crypto
seasonality) should show persistent calendar effects.

Construction:
  ret = close.pct_change()
  for each date t in month m: seasonal = mean(ret over same month m in the
  prior 3 years, using only returns whose dates are >= t-4y and < t-1y).
Simple version: mean of daily pct_change on dates d where d.month == m and
(d < t - 365d) and (d >= t - 4*365d).

This is likely orthogonal to momentum (prior-year same-month returns vs recent
returns) and to price-location.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   coverage, turnover_rank, fwd_returns,
                                   ic_series, validate_factor, load_library_panels,
                                   max_library_corr, IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()


def seasonal_month_hist(c, v, o, h, l, m, years=3, min_obs=30):
    """Per-asset mean daily return in the same calendar month over prior years."""
    r = c.pct_change()
    idx = c.index
    out = pd.Series(np.nan, index=idx)
    # precompute per-year-month means over the FULL history, then map
    df = pd.DataFrame({"r": r, "y": idx.year, "m": idx.month}).dropna()
    grp = df.groupby(["y", "m"])["r"].mean()
    for dt in idx:
        yy, mm = dt.year, dt.month
        prior = grp.loc[(slice(None), mm)]
        prior = prior[[y for y, _ in prior.index if (yy - years) <= y < yy]]
        if len(prior) >= 1 and prior.notna().sum() >= 1:
            out.loc[dt] = prior.mean()
    return out


def seasonal_win(prior):
    """Weighted same-month seasonality: more recent years weighted 3/2/1."""
    pass


print(">>> calendar seasonality screen", flush=True)
lib = load_library_panels()
print(f"library panels loaded: {list(lib.keys())}", flush=True)

for name, prm in [
    ("seas_m3", dict(years=3)),
    ("seas_m4", dict(years=4)),
    ("seas_m2", dict(years=2)),
]:
    panel = factor_panel(seasonal_month_hist, close, vol, open_, high, low, macro, **prm)
    res = validate_factor(seasonal_month_hist, close, vol, open_, high, low, macro, **prm)
    res["max_abs_library_correlation"] = max_library_corr(panel, lib)
    ic = res["ic"]; icir = res["icir"]
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"=== {name} (years={prm['years']}) ===", flush=True)
    print(f"  ic={ic:.4f} icir={icir:.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']} cov={res['coverage_asset_days']:.3f}/{res['coverage_dates_ge8']:.2f} "
          f"to={res['turnover_10d_rank']:.2f}", flush=True)
    print(f"  decay={res['decay_ic_by_horizon']}", flush=True)
    print(f"  max_abs_library_correlation={res['max_abs_library_correlation']:.4f}", flush=True)
    ic10 = ic_series(panel, fwd_returns(close, 10))
    regs = {"2020": ("2020-01-01", "2020-12-31"), "2021": ("2021-01-01", "2021-12-31"),
            "2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"),
            "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
            "2026H1": ("2026-01-01", "2026-06-30")}
    print("  regime IC (h=10):", flush=True)
    for rname, (a, b) in regs.items():
        sub = ic10.loc[(ic10.index >= a) & (ic10.index <= b)]
        if len(sub):
            print(f"    {rname}: ic={sub.mean():.4f} n={len(sub)}", flush=True)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("done", flush=True)
