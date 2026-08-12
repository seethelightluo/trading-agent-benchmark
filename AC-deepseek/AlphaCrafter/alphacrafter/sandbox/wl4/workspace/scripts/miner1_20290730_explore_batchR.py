"""miner1 2029-07-30 factor exploration batch R.
Motivation: current tape is defensive (VIX stress, BTC multi-block downtrend, broad
equity/rate selloff). Explore asymmetry / relative-strength / regime-conditional
candidates complementary to library (dn_mkt_beta, rate_beta_cn10y, vol_adj_mom_accel).
"""
import sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, ret_panel, forward_returns, rank_ic_series,
    summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    library_signals, max_library_corr, full_eval,
)

panels = load_panels(days=3000)
px = close_panel(panels)
rets = px.pct_change()
assets = px.columns.tolist()
print("assets:", len(assets), "dates:", len(px), px.index.min().date(), "->", px.index.max().date())

# volume coverage check
vol_ok = {}
for a in assets:
    df = panels.get(a)
    if df is not None and "volume" in df.columns:
        v = df["volume"].dropna()
        vol_ok[a] = (len(v) / len(df), float(v.iloc[-1]) if len(v) else 0.0)
print("volume coverage:", {k: round(v[0], 2) for k, v in vol_ok.items()})

macro = {m: panels[m]["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"] if m in panels}
vix = macro["VIX"]
vix_ret = vix.pct_change()

# ---- library signals for correlation audit ----
lib = library_signals(panels, closes=px, rets=rets, vix=vix)
# add current effective factors
mkt = rets.mean(axis=1)
dn_x = mkt.clip(upper=0.0)
def rolling_beta_fast(y, x, win, minp):
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    b = z["y"].rolling(win, min_periods=minp).cov(z["x"]) / z["x"].rolling(win, min_periods=minp).var()
    return b
lib["dn_mkt_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), dn_x, 60, 40))
cn10y_ret = px["CN10Y"].pct_change()
lib["rate_beta_cn10y_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), cn10y_ret, 60, 40))
def vol_adj_mom_accel(c, fast=20, slow=60, vol_win=20):
    r = c.pct_change()
    return (c / c.shift(fast) - 1.0 - (c / c.shift(slow) - 1.0)) / r.rolling(vol_win).std()
lib["vol_adj_mom_accel_20x60"] = px.apply(vol_adj_mom_accel)

# ---- candidate factors ----
cand = {}

# 1. upside/downside capture asymmetry over 60d
pos_r = rets.where(rets > 0, 0.0)
neg_r = rets.where(rets < 0, 0.0)
up_sum = pos_r.rolling(60, min_periods=40).sum()
dn_sum = neg_r.rolling(60, min_periods=40).sum()
cand["updown_capture_60d"] = up_sum / dn_sum.abs().replace(0, np.nan)

# 2. range position (stochastic): where close sits in 20d high-low range
hl = pd.DataFrame(index=px.index, columns=assets)
for a in assets:
    df = panels[a]
    hh = df["high"].rolling(20, min_periods=15).max()
    ll = df["low"].rolling(20, min_periods=15).min()
    hl[a] = (df["close"] - ll) / (hh - ll).replace(0, np.nan)
cand["range_pos_20d"] = hl

# 3. trend consistency: up-day fraction over 60d (count-based momentum)
cand["upday_frac_60d"] = (rets > 0).astype(float).rolling(60, min_periods=40).mean()

# 4. downside vol / upside vol asymmetry over 20d
cand["down_up_vol_20d"] = neg_r.rolling(20, min_periods=15).std() / pos_r.rolling(20, min_periods=15).std().replace(0, np.nan)

# 5. crypto-lead beta: beta of asset returns to BTC returns over 60d (risk-sentiment leadership)
btc_ret = px["BTC"].pct_change()
cand["btc_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), btc_ret, 60, 40))

# 6. gold-beta (haven rotation) over 60d
xau_ret = px["XAU"].pct_change()
cand["xau_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), xau_ret, 60, 40))

# 7. VIX-regime conditional momentum: 20d mom scaled down when VIX above its 60d median
vix_hi = (vix > vix.rolling(60, min_periods=40).median()).astype(float)
mom20 = px / px.shift(20) - 1.0
cand["vix_cond_mom_20d"] = mom20 * (1.0 - 0.6 * vix_hi.reindex(px.index).ffill())

# 8. idiosyncratic relative strength: 20d return minus cross-sectional median, / cross-sectional std
cs_med = mom20.median(axis=1)
cs_std = mom20.std(axis=1)
cand["rel_strength_20d"] = (mom20 - cs_med.values.reshape(-1, 1)) / cs_std.values.reshape(-1, 1).replace(0, np.nan)

# 9. drawdown depth normalized by 60d max drawdown (recovery position)
dd = px / px.rolling(60, min_periods=40).max() - 1.0
max_dd_60 = dd.rolling(60, min_periods=40).min()
cand["dd_recovery_60d"] = (dd - max_dd_60).clip(lower=0) / (max_dd_60.abs().replace(0, np.nan) + 1e-9)

# 10. volume-confirmed momentum (20d mom * volume ratio 20/60) if volume available
vol_ratio = pd.DataFrame(index=px.index, columns=assets)
for a in assets:
    df = panels[a]
    if "volume" in df.columns and df["volume"].dropna().shape[0] > 100:
        v = df["volume"].astype(float)
        vol_ratio[a] = v.rolling(20, min_periods=10).mean() / v.rolling(60, min_periods=30).mean()
    else:
        vol_ratio[a] = np.nan
cand["vol_confirm_mom_20d"] = mom20 * vol_ratio

# 11. momentum risk-adjusted at 60d (trend efficiency): |60d return| / sum |daily ret|
cand["trend_eff_60d"] = (px / px.shift(60) - 1.0).abs() / rets.abs().rolling(60, min_periods=40).sum()

# ---- forward returns ----
fwd = {h: px.shift(-h) / px - 1.0 for h in [1, 2, 3, 5, 10, 20]}

print("\n=== candidate screen (h=10 IC / ICIR; gate |IC|>=0.007 & |ICIR|>=0.084) ===")
results = {}
for name, fv in cand.items():
    ic10 = rank_ic_series(fv, fwd[10])
    s = summarize_ic(ic10, name=name)
    s.update(coverage_metrics(fv, min_valid=8))
    s["turnover_10d_rank"] = turnover_rank(fv, 10)
    s["decay_ic_by_horizon"] = decay_profile(fv, px, (1, 2, 3, 5, 10, 20), 8, 1)
    mcorr, mfid, _ = max_library_corr(fv, lib)
    s["max_abs_library_correlation"] = round(mcorr, 4)
    s["max_corr_factor"] = mfid
    gate_ic = abs(s.get("ic", 0)) >= 0.007
    gate_icir = abs(s.get("icir", 0)) >= 0.084
    s["GATE"] = "PASS" if (gate_ic and gate_icir) else "fail"
    results[name] = s
    print(json.dumps({k: s.get(k) for k in ["name", "n_ic_dates", "ic", "icir", "ic_hit_ratio",
                                             "coverage_asset_days", "coverage_dates_ge8",
                                             "turnover_10d_rank", "max_abs_library_correlation",
                                             "max_corr_factor", "GATE"]}, default=str))

# full detail for passing candidates
print("\n=== PASSING CANDIDATES full metrics ===")
for name, s in results.items():
    if s.get("GATE") == "PASS":
        print(json.dumps({k: s.get(k) for k in ["name", "n_ic_dates", "ic", "icir", "ic_std",
                                                 "ic_hit_ratio", "coverage_asset_days", "coverage_dates_ge8",
                                                 "turnover_10d_rank", "decay_ic_by_horizon",
                                                 "max_abs_library_correlation", "max_corr_factor"]}, default=str))

# regime snapshot
print("\n=== regime snapshot ===")
mkt_ret = rets.mean(axis=1)
for w in [20, 60]:
    r = (1 + mkt_ret).rolling(w).apply(np.prod, raw=True) - 1
    v = mkt_ret.rolling(w).std() * np.sqrt(252)
    print(f"mkt(live) {w:3d}d cum: {r.iloc[-1]*100:+.2f}%  vol_ann: {v.iloc[-1]*100:.1f}%")
print("VIX last:", round(float(vix.iloc[-1]), 2), " 60d ago:", round(float(vix.iloc[-61]), 2))
print("DXY last:", round(float(macro['DXY'].iloc[-1]), 2), " 60d ago:", round(float(macro['DXY'].iloc[-61]), 2))
print("USDJPY last:", round(float(macro['USDJPY'].iloc[-1]), 2), " 60d ago:", round(float(macro['USDJPY'].iloc[-61]), 2))

json.dump({k: {kk: (str(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
           for k, v in results.items()},
          open("scripts/_miner1_20290730_batchR_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/_miner1_20290730_batchR_results.json")
