"""miner_2 2026-07-30 -- calendar seasonality factor screen (fixed).

Idea: per-asset historical average return in the current calendar month,
computed from PRIOR years only (rolling same-month window shifted back in
time to avoid lookahead). Assets with persistent calendar effects (commodity
demand seasons, equity Jan/Dec effects, crypto seasonality) should show up.

Variants:
  seas_dmean_yK : mean of daily pct_change on same-month dates in prior K years
  seas_mret_yK  : mean of full-month return (m-1 last close -> m last close)
                  in prior K years
  seas_dmean_w  : weighted daily-mean (more recent years weighted 3/2/1)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   ic_series, fwd_returns, validate_factor,
                                   load_library_panels, max_library_corr,
                                   IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()


def _prior_year_month_means(r, years, weights=None):
    """Map date -> mean daily return in same calendar month over prior years.

    Returns (Series aligned to r.index). For date t (year yy, month mm):
    mean of daily returns on dates d with d.month==mm and yy-years <= d.year < yy.
    Optionally weight each year's contribution (most recent weight largest).
    """
    idx = r.index
    out = pd.Series(np.nan, index=idx)
    df = pd.DataFrame({"r": r.values, "y": idx.year, "m": idx.month}).dropna()
    grp = df.groupby(["y", "m"])["r"].mean()
    for mm in range(1, 13):
        try:
            by_year = grp.loc[(slice(None), mm)]
        except Exception:
            by_year = grp[grp.index.get_level_values("m") == mm]
        if by_year.empty:
            continue
        yrs = by_year.index if isinstance(by_year.index, pd.Index) else by_year.index
        # vectorised: for each date in this month, take mean over prior years
        mask = np.asarray(idx.month == mm)
        if not mask.any():
            continue
        yy = idx[mask].year.values
        w = np.ones(len(yrs)) if weights is None else weights
        vals = np.full(len(yy), np.nan)
        for i, y0 in enumerate(yy):
            sel = by_year[[y for y in yrs if (y0 - years) <= y < y0]]
            if len(sel):
                ww = np.array([w[yrs.get_loc(y)] for y in sel.index]) if weights is not None else np.ones(len(sel))
                vals[i] = float(np.average(sel.values, weights=ww))
        out.iloc[mask] = vals
    return out


def seas_dmean(c, v, o, h, l, macro, years=3):
    """Mean daily return in same calendar month over prior K years."""
    return _prior_year_month_means(c.pct_change(), years)


def seas_mret(c, v, o, h, l, macro, years=3):
    """Mean full-month return in same calendar month over prior K years."""
    mc = c.resample("ME").last()
    mret = mc.pct_change()
    s = _prior_year_month_means(mret, years)
    # map month-level estimates back to daily dates
    out = pd.Series(np.nan, index=c.index)
    for dt in c.index:
        key = pd.Timestamp(dt.year, dt.month, 1)
        if key in s.index and np.isfinite(s.loc[key]):
            out.loc[dt] = s.loc[key]
    return out


def seas_dmean_w(c, v, o, h, l, macro, years=3):
    """Weighted (3/2/1 most-recent) mean daily return, same month, prior years."""
    r = c.pct_change()
    idx = r.index
    df = pd.DataFrame({"r": r.values, "y": idx.year, "m": idx.month}).dropna()
    grp = df.groupby(["y", "m"])["r"].mean()
    out = pd.Series(np.nan, index=idx)
    for mm in range(1, 13):
        try:
            by_year = grp.loc[(slice(None), mm)]
        except Exception:
            by_year = grp[grp.index.get_level_values("m") == mm]
        if by_year.empty:
            continue
        yrs = list(by_year.index)
        mask = np.asarray(idx.month == mm)
        if not mask.any():
            continue
        yy = idx[mask].year.values
        vals = np.full(len(yy), np.nan)
        for i, y0 in enumerate(yy):
            sel = by_year[[y for y in yrs if (y0 - years) <= y < y0]]
            if len(sel):
                ww = np.arange(len(sel), 0, -1).astype(float)  # most recent = largest
                vals[i] = float(np.average(sel.values, weights=ww))
        out.iloc[mask] = vals
    return out


print(">>> calendar seasonality screen v2", flush=True)
lib = load_library_panels()
print(f"library panels loaded: {list(lib.keys())}", flush=True)

REGIONS = {"2020": ("2020-01-01", "2020-12-31"), "2021": ("2021-01-01", "2021-12-31"),
           "2022": ("2022-01-01", "2022-12-31"), "2023": ("2023-01-01", "2023-12-31"),
           "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
           "2026H1": ("2026-01-01", "2026-06-30")}

for name, fn, prm in [
    ("seas_dmean_m3", seas_dmean, dict(years=3)),
    ("seas_dmean_m4", seas_dmean, dict(years=4)),
    ("seas_dmean_m2", seas_dmean, dict(years=2)),
    ("seas_mret_m3", seas_mret, dict(years=3)),
    ("seas_mret_m4", seas_mret, dict(years=4)),
    ("seas_dmean_w_m3", seas_dmean_w, dict(years=3)),
]:
    panel = factor_panel(fn, close, vol, open_, high, low, macro, **prm)
    res = validate_factor(fn, close, vol, open_, high, low, macro, **prm)
    res["max_abs_library_correlation"] = max_library_corr(panel, lib)
    ic, icir = res["ic"], res["icir"]
    ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
    print(f"=== {name} (years={prm['years']}) ===", flush=True)
    print(f"  ic={ic:.4f} icir={icir:.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']} cov={res['coverage_asset_days']:.3f}/{res['coverage_dates_ge8']:.2f} "
          f"to={res['turnover_10d_rank']:.2f}", flush=True)
    print(f"  decay={res['decay_ic_by_horizon']}", flush=True)
    print(f"  max_abs_library_correlation={res['max_abs_library_correlation']:.4f}", flush=True)
    ic10 = ic_series(panel, fwd_returns(close, 10))
    print("  regime IC (h=10):", flush=True)
    for rname, (a, b) in REGIONS.items():
        sub = ic10.loc[(ic10.index >= a) & (ic10.index <= b)]
        if len(sub):
            print(f"    {rname}: ic={sub.mean():.4f} n={len(sub)}", flush=True)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}", flush=True)
print("done", flush=True)
