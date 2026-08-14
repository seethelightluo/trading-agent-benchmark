"""miner_1 factor exploration batch - 2035-06-25 cycle.

Explore novel cross-asset factor candidates on the 15-asset tradable universe
(no lookahead; data visible through the previous completed trading day).
Candidates focus on:
  - time-series trend/mean-reversion structure (distance-from-high, TS mom, RSI)
  - risk-adjusted carry (return per downside deviation, cvar ratio)
  - return distribution shape (skew, kurtosis)
  - macro sensitivity (DXY beta, USDJPY beta)
  - bond-equity regime interaction (US10Y momentum cross)
Each candidate is evaluated with rank IC at h=10 and the shared admission gates
(|IC|>=0.0070, |ICIR|>=0.0840) over the full sample and a recent 2y window.
"""
import sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series, summarize_ic,
    coverage_metrics, turnover_rank, decay_profile, max_library_corr,
)

t0 = time.time()
panels = load_panels(days=3500)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape {closes.shape}  last date {closes.index.max()}  ({time.time()-t0:.1f}s)", flush=True)

# macro panels
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
usdjpy = panels["USDJPY"]["close"].astype(float) if "USDJPY" in panels else None
vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
us10y = closes["US10Y"]
cn10y = closes["CN10Y"]

sig = {}

# --- 1. Distance from 60d high (mean-reversion candidate, neg expected) ---
sig["dist_high_60d"] = closes / closes.rolling(60).max() - 1.0

# --- 2. Time-series momentum 60d (trend continuation, pos expected) ---
sig["ts_mom_60d"] = closes / closes.shift(60) - 1.0

# --- 3. RSI 14 (mean-reversion in cross-section, neg expected) ---
delta = closes.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
sig["rsi_14d"] = 100.0 - 100.0 / (1.0 + rs)

# --- 4. Return per downside deviation (risk-adjusted carry, pos) ---
downside = rets.clip(upper=0)
sig["carry_dd_20d"] = (closes / closes.shift(20) - 1.0) / downside.rolling(20).std()

# --- 5. Downside vol ratio: downside vol / total vol (safety, pos: low ratio wins) ---
sig["downside_vol_ratio_20d"] = downside.rolling(20).std() / rets.rolling(20).std()

# --- 6. Rolling skewness 40d (pos expected: positive skew assets) ---
sig["skew_40d"] = rets.rolling(40).skew()

# --- 7. DXY beta 60d (neg expected: low USD-beta wins) ---
if dxy is not None:
    dxy_ret = dxy.pct_change()
    beta = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        beta[a] = z["a"].rolling(60).cov(z["d"]) / z["d"].rolling(60).var()
    sig["dxy_beta_60d"] = -beta

# --- 8. USDJPY beta 60d (neg expected: low JPY-beta wins in risk regimes) ---
if usdjpy is not None:
    jpy_ret = usdjpy.pct_change()
    beta = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), jpy_ret.rename("j")], axis=1).dropna()
        beta[a] = z["a"].rolling(60).cov(z["j"]) / z["j"].rolling(60).var()
    sig["usdjpy_beta_60d"] = -beta

# --- 9. Bond-equity regime: US10Y 20d momentum cross (defensive when yields rise) ---
us10y_mom = us10y.pct_change(20)
sig["yield_mom_cross_20d"] = -us10y_mom.to_frame("y").join(
    pd.DataFrame(index=closes.index), how="right")[ "y"]  # placeholder fix below

# --- 10. Vol-of-vol 20x60 (low vol-of-vol wins, neg expected) ---
sig["vol_of_vol_20x60"] = rets.rolling(20).std().rolling(60).std()

# --- 11. 60d high-low position (range position, pos expected) ---
hh = closes.rolling(60).max()
ll = closes.rolling(60).min()
sig["hl_pos_60d"] = (closes - ll) / (hh - ll).replace(0, np.nan)

# --- 12. Momentum ratio 20/60 (acceleration, pos expected) ---
sig["mom_ratio_20_60"] = (closes / closes.shift(20) - 1.0) / (closes / closes.shift(60) - 1.0).abs()

# --- 9 (fixed): US10Y yield momentum cross ---
yield_mom = us10y.pct_change(20)
sig["yield_mom_cross_20d"] = -yield_mom.to_frame("ym").join(rets, how="right")["ym"]

# --- 13. 20d realized vol z-score vs 60d (low current vol wins, neg expected) ---
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
sig["vol_z_20x60"] = (vol20 - vol60) / vol60.replace(0, np.nan)

# --- 14. Drawdown 20d (max peak-to-trough over 20d, neg expected: less drawdown wins) ---
roll_max20 = closes.rolling(20).max()
sig["dd_20d"] = closes / roll_max20 - 1.0

# --- Reconstruct library factors for correlation ---
# vol_adj_mom_accel_20x60
lib_mom = (closes / closes.shift(20) - 1.0 - (closes / closes.shift(60) - 1.0)) / rets.rolling(20).std()
# dn_mkt_beta_60d
mkt = rets.mean(axis=1)
down_mkt = mkt.where(mkt < 0)
lib_beta = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down_mkt.rename("m")], axis=1).dropna()
    lib_beta[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
# rate_beta_cn10y_60d
cn_ret = cn10y.pct_change()
lib_rate = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn_ret.rename("c")], axis=1).dropna()
    lib_rate[a] = z["a"].rolling(60).cov(z["c"]) / z["c"].rolling(60).var()

library = {
    "vol_adj_mom_accel_20x60": lib_mom,
    "dn_mkt_beta_60d": lib_beta,
    "rate_beta_cn10y_60d": lib_rate,
}

# --- Evaluation ---
expected = {
    "dist_high_60d": -1, "ts_mom_60d": 1, "rsi_14d": -1, "carry_dd_20d": 1,
    "downside_vol_ratio_20d": 1, "skew_40d": 1, "dxy_beta_60d": 1,
    "usdjpy_beta_60d": 1, "yield_mom_cross_20d": 1, "vol_of_vol_20x60": -1,
    "hl_pos_60d": 1, "mom_ratio_20_60": 1, "vol_z_20x60": -1, "dd_20d": -1,
}

full = closes.index >= "2020-01-01"
recent = closes.index >= "2033-06-01"

results = []
for name, f in sig.items():
    f = f.reindex(closes.index)
    row = {"factor": name, "expected_sign": expected.get(name)}
    for label, mask in [("full", full), ("recent2y", recent)]:
        fp = f[mask]
        cp = closes[mask]
        ics = rank_ic_series(fp, forward_returns(cp, 10), 8)
        m = summarize_ic(ics, expected.get(name))
        row[f"ic_{label}"] = m["ic"]
        row[f"icir_{label}"] = m["icir"]
        row[f"n_{label}"] = m["n_ic_dates"]
    # decay + coverage + turnover on full
    fp, cp = f[full], closes[full]
    ics = rank_ic_series(fp, forward_returns(cp, 10), 8)
    m = summarize_ic(ics, expected.get(name))
    m.update(coverage_metrics(fp))
    m["turnover_10d_rank"] = turnover_rank(fp, 10)
    m["decay"] = decay_profile(fp, cp, (1, 2, 3, 5, 10, 20), 8, expected.get(name))
    corr, key = max_library_corr(fp, library)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    row.update(m)
    results.append(row)

res = pd.DataFrame(results)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print("\n=== CANDIDATE EVALUATION (full sample 2020-01-01 .. %s) ===" % closes.index.max().date())
print(res[["factor", "expected_sign", "ic_full", "icir_full", "n_full",
           "ic_recent2y", "icir_recent2y", "n_recent2y", "ic_hit_ratio",
           "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
           "max_abs_library_correlation", "max_corr_factor"]].to_string(index=False))

print("\n=== DECAY PROFILES (full) ===")
for _, r in res.iterrows():
    print(f"{r['factor']:28s} {json.dumps(r['decay'])}")

print(f"\ntotal runtime {time.time()-t0:.1f}s")
