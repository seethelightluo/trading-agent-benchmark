"""miner_1 2031-12-25: explore novel factor candidates (batch A).

Visible data through 2031-12-24. Candidate families:
1. Cross-asset exposure betas (DXY, XAU, WTI, COPPER, NDX) - unconditional rolling betas
   that capture regime exposure beyond the conditional SPX/USDJPY betas already in library.
2. Return-shape factors: up-day frequency, up/down magnitude asymmetry, skew, kurtosis.
3. Volatility-structure factors: 5d/60d vol ratio (short-term vol spike).
4. Trend-acceleration: mom_60 - mom_20 (short vs medium horizon divergence).
Each candidate validated at h=10 (admission horizon) with IC/ICIR gates (|IC|>=0.007, |ICIR|>=0.084).
"""
import sys
sys.path.insert(0, "scripts")
from miner_1_20311225_lib import (
    TRADABLES, load_panel, macro_series, per_asset, forward_returns, compute_ic,
    validate_factor, regime_split_ic, report, build_active_library, panel_rank_corr,
)
import numpy as np
import pandas as pd

panel = load_panel()
close = panel
print(f"panel dates: {panel.index[0].date()} -> {panel.index[-1].date()}, n={len(panel)}")

# ---- macro series ----
dxy = macro_series("DXY").pct_change()
xau = close["XAU"].dropna().pct_change()
wti = close["WTI"].dropna().pct_change()
copper = close["COPPER"].dropna().pct_change()
ndx = close["NDX"].dropna().pct_change()
spx = close["SPX"].dropna().pct_change()

def roll_beta_panel(asset_ret, anchor_ret, win, minp=None):
    """Rolling beta of asset returns on anchor returns, per asset, reindexed to panel."""
    out = {}
    for a in close.columns:
        s = close[a].dropna()
        ar = s.pct_change()
        df = pd.concat([ar.rename("a"), anchor_ret.reindex(ar.index).rename("m")], axis=1).dropna()
        b = df["a"].rolling(win).cov(df["m"]) / df["m"].rolling(win).var()
        if minp:
            b[df["m"].rolling(win).count() < minp] = np.nan
        out[a] = b.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)

def roll_vol(s, win):
    return s.pct_change().rolling(win).std()

# ---- candidate factors ----
factors = {}

# 1. Unconditional betas to macro/asset anchors
factors["dxy_beta_60"] = roll_beta_panel(close, dxy, 60, 30)
factors["xau_beta_60"] = roll_beta_panel(close, xau, 60, 30)
factors["wti_beta_60"] = roll_beta_panel(close, wti, 60, 30)
factors["copper_beta_60"] = roll_beta_panel(close, copper, 60, 30)
factors["ndx_beta_60"] = roll_beta_panel(close, ndx, 60, 30)

# 2. Return-shape factors
def upday_freq(s, win=60):
    r = s.pct_change()
    return r.rolling(win).apply(lambda x: (x > 0).mean(), raw=True)
factors["upday_freq_60"] = per_asset(close, upday_freq, 60)

def updown_asym(s, win=60):
    r = s.pct_change()
    def f(x):
        up = x[x > 0]
        dn = x[x < 0]
        if len(up) == 0 or len(dn) == 0:
            return np.nan
        return float(up.mean() / abs(dn.mean()))
    return r.rolling(win).apply(f, raw=True)
factors["updown_asym_60"] = per_asset(close, updown_asym, 60)

def skew_60(s, win=60):
    return s.pct_change().rolling(win).skew()
factors["skew_60"] = per_asset(close, skew_60, 60)

def kurt_60(s, win=60):
    return s.pct_change().rolling(win).kurt()
factors["kurt_60"] = per_asset(close, kurt_60, 60)

# 3. Volatility structure
def vol_ratio_5_60(s):
    r = s.pct_change()
    v5 = r.rolling(5).std()
    v60 = r.rolling(60).std()
    return v5 / v60 - 1.0
factors["vol_ratio_5_60"] = per_asset(close, vol_ratio_5_60)

# 4. Trend acceleration
def mom_diff(s, short=20, long=60):
    m_s = s / s.shift(short) - 1.0
    m_l = s / s.shift(long) - 1.0
    return m_s - m_l
factors["mom_accel_20_60"] = per_asset(close, mom_diff, 20, 60)

# ---- validation ----
library = build_active_library(panel)
fwd_cache = {}
results = {}
print("\n=== candidate validation (h=10 admission) ===")
for name, fp in factors.items():
    m = validate_factor(fp, panel, library=library, fwd_cache=fwd_cache)
    results[name] = m
    passed = report(name, m)
    if passed:
        reg = regime_split_ic(fp, forward_returns(panel, 10))
        print("   regime:", reg)

print("\n=== summary table ===")
for name, m in results.items():
    print(f"{name}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"cov_asset={m['coverage_asset_days']:.3f} cov_dates={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']} maxlibcorr={m['max_abs_library_correlation']}")

import json
with open("scripts/miner_1_20311225_batchA_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner_1_20311225_batchA_results.json")
