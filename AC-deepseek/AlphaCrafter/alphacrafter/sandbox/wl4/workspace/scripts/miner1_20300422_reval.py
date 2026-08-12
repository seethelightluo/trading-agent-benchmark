"""miner1 2030-04-22 factor exploration batch S.
Regime: defensive tilt with commodity rebound (COPPER/WTI strong, BTC/CN10Y frozen
stale, rates unwinding). Re-validate library factors + test new candidates.
Gate: |IC|>=0.0070 and |ICIR|>=0.0840 at h=10, min 8 valid instruments/date.
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
# also historical rejected/evicted factors for correlation audit
mom20 = px / px.shift(20) - 1.0
lib["mom_20d"] = mom20
lib["rsi14"] = px.apply(lambda c: c.rolling(14).apply(lambda w: 100 - 100/(1 + (w.diff().clip(lower=0).mean()/w.diff().clip(upper=0).abs().mean())), raw=False) if False else np.nan)
lib["vol_price_corr_20"] = px.apply(lambda c: c.pct_change().rolling(20).corr(c.pct_change().abs().rolling(20).mean()))

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

print("\n=== 1) LIBRARY FACTOR RE-VALIDATION (full window through 2030-04-19, h=10) ===")
lib_results = {}
for name, fv in lib.items():
    s = eval_factor(fv, name, expected_sign=1)
    lib_results[name] = s
    # recent-window drift checks
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

json.dump(lib_results, open("scripts/_miner1_20300422_lib_reval.json", "w"), indent=1, default=str)
print("saved scripts/_miner1_20300422_lib_reval.json")
