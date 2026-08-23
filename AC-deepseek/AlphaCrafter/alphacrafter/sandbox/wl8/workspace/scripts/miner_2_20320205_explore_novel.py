"""miner_2 2032-02-05: Explore novel cross-asset factor ideas
Current date: 2032-02-05, visible_through: 2032-02-04
Regime: High-vol divergent, fallback ensemble operative, 0 active factors in library
Goal: Discover new persisting factors beyond pure momentum/beta constructs
"""

import sys, os, json, math
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho, zscore_series)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index

print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)} px_last={px.index[-1].date()}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s)
    return (v.shift(-h) / v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s)
    return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def build(df):
    return df.sort_index().replace([np.inf, -np.inf], np.nan).astype(float)

def assess(name, factor_df, show_regimes=True):
    icd = cross_sectional_ic(factor_df, fwd)
    if len(icd) == 0:
        print(f"{name:35s} NO DATES (broadcast/constant)"); return None, icd
    st = ic_stats(icd)
    cov = (factor_df.notna() & fwd.notna()).mean().mean()
    line = (f"{name:35s} FULL IC={st['ic']:+.6f} ICIR={st['icir']:+.6f} hit={st['hit']:.3f} "
            f"n={st['n_dates']:5d} avg={st.get('avg_n',0):4.1f} cov={cov:.3f}")
    rmask = icd.index >= icd.index[-1] - pd.Timedelta(days=365)
    if rmask.any():
        ic365 = ic_stats(icd[rmask])
        line += f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    gate = abs(st['ic']) >= 0.0070 and abs(st['icir']) >= 0.0840
    line += f" | {'PASS' if gate else 'FAIL'}"
    print(line)
    if show_regimes:
        for lab, seg in regime_split(icd).items():
            print(f"    {lab}: [{seg[0]:+.4f},{seg[1]:+.4f},n={seg[2]}]")
    return st, icd

# =======================================================
# IDEA 1: Cross-sectional z-score of rolling return (relative momentum)
# Instead of raw momentum, rank within the 15-asset universe
# =======================================================
print("\n" + "=" * 80)
print("IDEA 1: Cross-sectional z-score of rolling returns (relative momentum)")
print("=" * 80)
for w in [5, 10, 20, 40]:
    raw = pd.DataFrame({s: retk(px[s], w) for s in WATCH})
    z = raw.rank(axis=1, pct=True).apply(lambda r: pd.Series(
        [(r.iloc[i] - 0.5) * 2 for i in range(len(r))], index=r.index), axis=1)
    z = build(z)
    assess(f'rel_mom_z_{w}d', z)

# =======================================================
# IDEA 2: Cross-sectional volatility ranking (low-vol anomaly)
# =======================================================
print("\n" + "=" * 80)
print("IDEA 2: Cross-sectional volatility ranking (low-vol anomaly in cross-asset)")
print("=" * 80)
for w in [10, 20, 60]:
    vol_raw = pd.DataFrame({s: rv(px[s], w) for s in WATCH})
    # Inverse rank: lower vol -> higher score
    irank = vol_raw.rank(axis=1, ascending=True, pct=True)
    f_lv = build(irank.apply(lambda r: pd.Series(
        [(r.iloc[i] - 0.5) * 2 for i in range(len(r))], index=r.index), axis=1))
    assess(f'low_vol_{w}d_rank', f_lv)

# =======================================================
# IDEA 3: Volume-confirmed momentum (price trend * volume ratio)
# =======================================================
print("\n" + "=" * 80)
print("IDEA 3: Volume-confirmed momentum (price trend weighted by volume ratio)")
print("=" * 80)
for w in [10, 20]:
    mom = pd.DataFrame({s: retk(px[s], w) for s in WATCH})
    # Volume ratio: recent vol vs longer-term vol
    def vol_ratio(s, short=5, long=60):
        v = vseries(s)
        vol_s = v.rolling(short).std()
        vol_l = v.rolling(long).std()
        return (vol_s / vol_l).reindex(INDEX)
    vr = pd.DataFrame({s: vol_ratio(px[s], 5, 60) for s in WATCH})
    # Volume-confirmed: momentum * volume ratio (high volume = more signal)
    f_vcm = build(mom * vr)
    assess(f'vol_conf_mom_{w}d', f_vcm)

# =======================================================
# IDEA 4: Distance from recent high (drawdown/reversal signal)
# =======================================================
print("\n" + "=" * 80)
print("IDEA 4: Distance from multi-period high (bounc