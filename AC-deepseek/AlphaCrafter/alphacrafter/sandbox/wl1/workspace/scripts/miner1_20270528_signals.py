"""miner_1 2027-05-28: compute library signals + candidate factor signals (v3).
Preprocessing: weekday-only rows, >=8 valid assets, ffill (same as prior cycles)
so union-calendar weekends/holidays do not wipe rolling windows. Then min_periods
on rolling windows as extra protection.
"""
import numpy as np
import pandas as pd
import pickle, math

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"].copy(); O = panel["open"].copy(); H = panel["high"].copy()
L = panel["low"].copy(); V = panel["vol"].copy(); M = panel["macro"].copy()

# ---- preprocessing: weekday-only, >=8 valid, ffill ----
wk = C.index.dayofweek < 5
C = C[wk]; O = O[wk]; H = H[wk]; L = L[wk]; V = V[wk]; M = M[wk]
keep = C.notna().sum(axis=1) >= 8
C = C[keep].ffill(); O = O[keep].ffill(); H = H[keep].ffill()
L = L[keep].ffill(); V = V[keep].ffill(); M = M[keep].ffill()

lr = C.pct_change(); lnC = np.log(C)
VIX = M["VIX"]; DXY = M["DXY"]
vix_ret = VIX.pct_change(); dxy_ret = DXY.pct_change()

sig = {}

def mp(w, frac=0.6):
    return max(int(w * frac), 3)

# ---------- library factors (14) ----------
sig["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for w in (1, 2, 3, 5):
    sig[f"nclv_{w}d"] = -(C - L.rolling(w).min()) / (H.rolling(w).max() - L.rolling(w).min())
sig["rev_1d"] = -(C / C.shift(1) - 1.0)
sig["rev_2d"] = -(C / C.shift(2) - 1.0)
sig["rev_3d"] = -(C / C.shift(3) - 1.0)
sig["rev_5d"] = -(C / C.shift(5) - 1.0)
sig["rev_1d_vs"] = -lnC.diff(1) / lr.rolling(20, min_periods=10).std()
sig["id_rev_1d"] = -(C / O - 1.0)
sig["nbody_1d"] = -((C - O) / (H - L))
vol20 = lr.rolling(20, min_periods=10).std()
sig["vol_of_vol20x60"] = vol20.rolling(60, min_periods=30).std()
beta60_vix = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
for c in C.columns:
    beta60_vix[c] = (lr[c].rolling(60, min_periods=30).cov(vix_ret) / vix_ret.rolling(60, min_periods=30).var())
vix_move20 = VIX / VIX.shift(20) - 1.0
sig["vix_beta_cond_60x20"] = -beta60_vix.mul(vix_move20, axis=0)

# ---------- candidate factors (novel) ----------
# 1. Kaufman efficiency ratio
for w in (10, 20, 40, 60):
    net = (C / C.shift(w) - 1.0).abs()
    path = lr.abs().rolling(w, min_periods=mp(w)).sum()
    sig[f"er_{w}"] = net / path
# 2. relative (panel-mean-adjusted) momentum
for w in (20, 60, 120):
    m = C / C.shift(w) - 1.0
    sig[f"rel_mom_{w}"] = m.sub(m.mean(axis=1), axis=0)
# 3. up-day ratio (trend consistency)
for w in (20, 60):
    sig[f"up_ratio_{w}"] = (lr > 0).rolling(w, min_periods=mp(w)).mean()
# 4. rolling skewness
for w in (60, 120):
    sig[f"skew_{w}"] = lr.rolling(w, min_periods=mp(w)).skew()
# 5. drawdown depth from rolling max
for w in (60, 120):
    sig[f"dd_depth_{w}"] = C / C.rolling(w, min_periods=mp(w)).max() - 1.0
# 6. gap reversal: negative overnight gap
sig["gap_rev_1d"] = -(O / C.shift(1) - 1.0)
# 7. dollar beta conditional on DXY trend
beta60_dxy = pd.DataFrame(index=C.index, columns=C.columns, dtype=float)
for c in C.columns:
    beta60_dxy[c] = (lr[c].rolling(60, min_periods=30).cov(dxy_ret) / dxy_ret.rolling(60, min_periods=30).var())
dxy_move20 = DXY / DXY.shift(20) - 1.0
sig["dxy_beta_cond_60x20"] = -beta60_dxy.mul(dxy_move20, axis=0)
# 8. vol regime z-score: 20d vol vs its 60d mean
vol20m = vol20.rolling(60, min_periods=30).mean()
sig["vol_z_20x60"] = (vol20 - vol20m) / vol20.rolling(60, min_periods=30).std()
# 9. range efficiency: Parkinson vol / realized vol
park = np.sqrt((np.log(H / L) ** 2).rolling(20, min_periods=10).mean() / (4 * np.log(2)))
sig["park_ratio_20"] = park / vol20
# 10. momentum scaled by efficiency (trend-quality momentum)
for w in (20, 60):
    m = C / C.shift(w) - 1.0
    e = sig[f"er_{w}"]
    sig[f"momxer_{w}"] = m * e
# 11. volume z-score (check data first)
vz = (V - V.rolling(20, min_periods=10).mean()) / V.rolling(20, min_periods=10).std()
sig["volz_20"] = vz
# 12. 5d residual reversal (cross-section demeaned)
r5 = lr.rolling(5, min_periods=3).sum()
sig["rel_rev_5d"] = -(r5 - r5.mean(axis=1))

with open("scripts/miner1_20270528_signals.pkl", "wb") as fh:
    pickle.dump(sig, fh, protocol=4)
print("saved", len(sig), "signals; panel", C.shape, C.index.min().date(), "->", C.index.max().date())
for k in sig:
    print(f"{k:24s} valid_dates(>=8)={int(sig[k].notna().sum(axis=1).ge(8).sum())}")
