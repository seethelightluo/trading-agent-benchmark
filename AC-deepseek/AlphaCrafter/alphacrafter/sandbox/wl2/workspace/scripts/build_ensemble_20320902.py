"""Screener: build factor_ensemble.json for 2032-09-02 cycle (data through 2032-09-01)."""
import json

# Persisted validation metrics (miner-gate-passed set; admission unchanged this cycle - miner outputs empty)
factors = {
    "max_consec_gain_20": {"ic": 0.0682, "icir": 0.231, "turnover_10d_rank": 0.2318, "cat": "Momentum (20d max consecutive gain streak)"},
    "mom_180d_skip5":     {"ic": 0.0495, "icir": 0.125, "turnover_10d_rank": 0.268,   "cat": "Momentum (180d, skip-5 long-horizon trend)"},
    "downbeta_spx_60":    {"ic": 0.0752, "icir": 0.1871,"turnover_10d_rank": 0.072,   "cat": "Risk/Beta (60d down-market beta to SPX)"},
    "spx_corr60":         {"ic": 0.0558, "icir": 0.1556,"turnover_10d_rank": 0.0612,  "cat": "Risk/Beta (60d SPX rolling correlation)"},
    "range_pos_252":      {"ic": 0.0355, "icir": 0.107, "turnover_10d_rank": 0.1533,  "cat": "Trend/Quality (252d range position)"},
}

# Regime overlays decided 2032-09-02 (visible data through 2032-09-01)
overlays = {
    "max_consec_gain_20": 0.70,   # VIX +68%/20d whipsaw regime; SPX 7-streak yet -2.4%/20d, NDX 5-streak yet -7.8%/20d -> streaks forming in rolling-over names
    "mom_180d_skip5":     1.60,   # long-horizon trend cleanest separator (WTI +58%/60d, ETH +69%/60d @252d highs vs SOX -25%/180d, 000688 -16%/180d); robust to vol spikes
    "downbeta_spx_60":    0.45,   # ETH downbeta -3.69 fights factor (would de-weight top performer); 27/60 SPX down days raises downside relevance -> keep demoted
    "spx_corr60":         0.80,   # mean |corr| 0.095 low -> noisy; NDX/SOX negative corr correctly de-weights crash legs
    "range_pos_252":      1.25,   # trend-location separation still clean (ETH 0.99, WTI 0.99, SPX 0.85 vs SOX 0.15, 000688 0.23); slow factor robust in vol spikes
}

base = {f: abs(factors[f]["ic"]) * abs(factors[f]["icir"]) for f in factors}
sumb = sum(base.values())
w_base = {f: base[f] / sumb for f in factors}
w_ov = {f: w_base[f] * overlays[f] for f in factors}
sumo = sum(w_ov.values())
w_final = {f: w_ov[f] / sumo for f in factors}

print("base q:", {f: round(base[f], 6) for f in factors})
print("base w:", {f: round(w_base[f], 4) for f in factors})
print("overlaid w:", {f: round(w_ov[f], 4) for f in factors})
print("FINAL w:", {f: round(w_final[f], 4) for f in factors})
print("sum:", round(sum(w_final.values()), 6))
print("trend cluster (mom180+range):", round(w_final["mom_180d_skip5"] + w_final["range_pos_252"], 4))
print("beta cluster (downbeta+corr):", round(w_final["downbeta_spx_60"] + w_final["spx_corr60"], 4))
