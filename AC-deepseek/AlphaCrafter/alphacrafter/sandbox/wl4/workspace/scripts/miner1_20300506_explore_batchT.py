"""miner1 2030-05-06 factor exploration batch T.
Current date 2030-05-06, visible through 2030-05-03.
Regime per memory: commodity rebound (COPPER/WTI strong), rates unwinding,
defensive tilt; BTC/CN10Y frozen stale. Objective: re-validate library +
test NEW candidate factor families (macro-beta, drawup, downside-vol, skew,
pos-day ratio, crypto-beta, commodity-beta) at h=10.
Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840, min 8 valid instruments/date.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns, rank_ic_series,
    summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    library_signals, max_library_corr, full_eval,
)

panels = load_panels(days=3000)
px = close_panel(panels)
rets = px.pct_change()
assets = px.columns.tolist()
print("assets:", len(assets), "dates:", len(px), px.index.min().date(), "->", px.index.max().date())

macro = {m: panels[m]["close"].astype(float) for m in ["VIX", "DXY", "USDJPY", "USDCNY", "EURUSD"]}
vix = macro["VIX"]
vix_ret = vix.pct_change()
dxy_ret = macro["DXY"].pct_change()
usdjpy_ret = macro["USDJPY"].pct_change()
eurusd_ret = macro["EURUSD"].pct_change()
usdcny_ret = macro["USDCNY"].pct_change()

# ============ LIBRARY SIGNALS (re-validation) ============
lib = library_signals(panels, closes=px, rets=rets, vix=vix)
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
# historical/evicted factors for correlation audit
lib["mom_20d"] = px / px.shift(20) - 1.0
lib["vol_price_corr_20"] = px.apply(lambda c: c.pct_change().rolling(20).corr(c.pct_change().abs().rolling(20).mean()))
lib["vol_ratio_20_60"] = rets.rolling(20).std() / rets.rolling(60).std()

# ============ NEW CANDIDATES ============
print("\n=== building new candidates ===")
cand = {}

# T1: commodity beta 60d (beta of asset returns to commodity basket = mean(XAU,COPPER,WTI) returns)
comm_ret = px[["XAU", "COPPER", "WTI"]].pct_change().mean(axis=1)
cand["comm_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), comm_ret, 60, 40))

# T2: DXY beta 60d (USD sensitivity; risk-on/off)
cand["dxy_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), dxy_ret, 60, 40))

# T3: USDJPY beta 60d (global risk appetite proxy)
cand["usdjpy_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), usdjpy_ret, 60, 40))

# T4: crypto beta 60d (contagion to BTC)
cand["crypto_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), px["BTC"].pct_change(), 60, 40))

# T5: drawup_60d = close/rolling_max(close,60) - 1 (distance from high; trend persistence)
cand["drawup_60d"] = px / px.rolling(60, min_periods=40).max() - 1.0

# T6: downside_vol_ratio_20x60 = 20d downside semi-deviation / 20d total vol
def downside_ratio(c, win=20):
    r = c.pct_change()
    dn = r.clip(upper=0.0)
    dn_vol = (dn**2).rolling(win).mean().apply(np.sqrt)
    tot = r.rolling(win).std()
    return dn_vol / tot
cand["downside_vol_ratio_20x60"] = px.apply(downside_ratio)

# T7: skew_20d = rolling skewness of daily returns (crash-risk asymmetry)
cand["skew_20d"] = rets.rolling(20, min_periods=10).skew()

# T8: pos_ratio_60d = fraction of positive days over 60d (trend consistency)
cand["pos_ratio_60d"] = (rets > 0).rolling(60, min_periods=40).mean()

# T9: vol-scaled 60d momentum (mom60 / vol20) - distinct from accel (mom20-mom60)
cand["mom60_vol20"] = (px / px.shift(60) - 1.0) / rets.rolling(20).std()

# T10: eurusd beta 60d (re-test; was deprecated in library, check for re-emergence)
cand["eurusd_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), eurusd_ret, 60, 40))

# T11: XAU beta 60d (safe-haven sensitivity)
cand["xau_beta_60d"] = px.apply(lambda c: rolling_beta_fast(c.pct_change(), px["XAU"].pct_change(), 60, 40))

# T12: conditional rate beta: rate_beta_cn10y_60d * sign(CN10Y 20d momentum)
rb = px.apply(lambda c: rolling_beta_fast(c.pct_change(), cn10y_ret, 60, 40))
cand["cond_rate_beta_cn10y_60x20"] = rb * np.sign(cn10y_ret.rolling(20).mean())

fwd = {h: px.shift(-h) / px - 1.0 for h in [1, 2, 3, 5, 10, 20]}

def eval_factor(fv, name, expected_sign=1):
    ic10 = rank_ic_series(fv, fwd[10])
    s = summarize_ic(ic10, expected_sign)
    s.update(coverage_metrics(fv, min_valid=8))
    s["turnover_10d_rank"] = turnover_rank(fv, 10)
    s["decay_ic_by_horizon"] = decay_profile(fv, px, (1, 2, 3, 5, 10, 20), 8, expected_sign)
    mcorr, mfid, _ = max_library_corr(fv, lib)
    s["max_abs_library_correlation"] = round(mcorr, 4)
    s["max_corr_factor"] = mfid
    gate_ic = abs(s.get("ic", 0)) >= 0.007
    gate_icir = abs(s.get("icir", 0)) >= 0.084
    s["GATE"] = "PASS" if (gate_ic and gate_icir) else "fail"
    return s

print("\n=== 1) LIBRARY FACTOR RE-VALIDATION (full window through 2030-05-03, h=10) ===")
lib_results = {}
for name, fv in lib.items():
    if name in ("mom_10d_skip5", "mom_120d_skip5"):
        continue
    s = eval_factor(fv, name, expected_sign=1)
    lib_results[name] = s
    drift = {}
    for wname, n in [("r500", 500), ("r250", 250)]:
        sub = fv.iloc[-n:]
        ic = rank_ic_series(sub, fwd[10].loc[sub.index])
        if len(ic):
            drift[f"ic_{wname}"] = round(float(ic.mean()), 4)
            drift[f"icir_{wname}"] = round(float(ic.mean()/ic.std(ddof=1)), 4) if ic.std(ddof=1) > 0 else 0.0
    s.update(drift)
    print(json.dumps({k: s.get(k) for k in ["name", "n_ic_dates", "ic", "icir", "ic_hit_ratio",
                                             "coverage_asset_days", "coverage_dates_ge8",
                                             "turnover_10d_rank", "max_abs_library_correlation",
                                             "ic_r500", "icir_r500", "ic_r250", "icir_r250", "GATE"]}, default=str))

print("\n=== 2) NEW CANDIDATES (full window, h=10) ===")
cand_results = {}
for name, fv in cand.items():
    s = eval_factor(fv, name, expected_sign=1)
    cand_results[name] = s
    print(json.dumps({k: s.get(k) for k in ["name", "n_ic_dates", "ic", "icir", "ic_hit_ratio",
                                             "coverage_asset_days", "coverage_dates_ge8",
                                             "turnover_10d_rank", "max_abs_library_correlation",
                                             "max_corr_factor", "decay_ic_by_horizon", "GATE"]}, default=str))

print("\n=== 3) NEW CANDIDATES recent-window checks (last 500 / 250 days) ===")
for name, fv in cand.items():
    drift = {}
    for wname, n in [("r500", 500), ("r250", 250)]:
        sub = fv.iloc[-n:]
        ic = rank_ic_series(sub, fwd[10].loc[sub.index])
        if len(ic):
            drift[f"ic_{wname}"] = round(float(ic.mean()), 4)
            drift[f"icir_{wname}"] = round(float(ic.mean()/ic.std(ddof=1)), 4) if ic.std(ddof=1) > 0 else 0.0
    print(name, json.dumps(drift, default=str))

json.dump({"lib": {k: {kk: vv for kk, vv in v.items() if kk != "decay_ic_by_horizon"} for k, v in lib_results.items()},
           "cand": {k: {kk: vv for kk, vv in v.items() if kk != "decay_ic_by_horizon"} for k, v in cand_results.items()}},
          open("scripts/_miner1_20300506_batchT.json", "w"), indent=1, default=str)
print("\nsaved scripts/_miner1_20300506_batchT.json")
