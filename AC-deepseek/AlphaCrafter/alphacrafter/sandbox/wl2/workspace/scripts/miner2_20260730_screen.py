"""miner_2 factor screening 2026-07-30.
Evaluates a batch of candidate cross-asset factors at 10d horizon on the
15-asset tradable universe, visible window 2020-01-01..2026-07-29.
Admission gates: |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_20260730_factorlib import (
    load_panel, load_series, TRADABLE, OBS, load_visible_through,
    full_validate,
)

end = load_visible_through()
panel = load_panel(TRADABLE, "stock", end)
print(f"panel: {len(panel)} rows, {len(panel.columns)} assets, "
      f"{panel.index.min().date()}..{panel.index.max().date()}")

ret = panel.pct_change()
macro = load_panel(OBS, "index", end)
vix = macro["VIX"]
dxy = macro["DXY"]
usdcny = macro["USDCNY"]
usdjpy = macro["USDJPY"]
eurusd = macro["EURUSD"]

# helper: rolling beta of asset returns vs a macro return series
def rolling_beta(ret_df, macro_ret, win):
    out = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    for sym in ret_df.columns:
        z = pd.concat([ret_df[sym].rename("a"), macro_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        out[sym] = cov / var
    return out

# ---------------- candidate factors ----------------
cands = {}

# 1. drawdown_252: distance from 1-year high (mean-reversion candidate)
cands["drawdown_252"] = panel / panel.rolling(252).max() - 1.0

# 2. range_pos_60: position within 60d high-low range
def range_pos(win):
    hi = panel.rolling(win).max()
    lo = panel.rolling(win).min()
    return (panel - lo) / (hi - lo).replace(0, np.nan)
cands["range_pos_60"] = range_pos(60)

# 3. skew_60: rolling skewness of 60d returns (crash-risk asymmetry)
cands["skew_60"] = ret.rolling(60).skew()

# 4. vol_regime_10x60: vol acceleration
cands["vol_regime_10x60"] = ret.rolling(10).std() / ret.rolling(60).std() - 1.0

# 5. dxy_beta_cond_60x20: conditional USD-strength sensitivity
cands["dxy_beta_cond_60x20"] = rolling_beta(ret, dxy.pct_change(), 60) * (dxy / dxy.shift(20) - 1.0)

# 6. usdcny_beta_cond_60x20: conditional China-risk sensitivity
cands["usdcny_beta_cond_60x20"] = rolling_beta(ret, usdcny.pct_change(), 60) * (usdcny / usdcny.shift(20) - 1.0)

# 7. us10y_beta_cond_60x20: conditional rates sensitivity
cands["us10y_beta_cond_60x20"] = rolling_beta(ret, panel["US10Y"].pct_change(), 60) * (panel["US10Y"] / panel["US10Y"].shift(20) - 1.0)

# 8. crypto_beta_60: risk-on beta to BTC
cands["crypto_beta_60"] = rolling_beta(ret, ret["BTC"], 60)

# 9. mom_sharpe_60: risk-adjusted 60d momentum
cands["mom_sharpe_60"] = (panel / panel.shift(60) - 1.0) / ret.rolling(20).std().clip(lower=1e-4)

# 10. downside_asym_60: downside vol / upside vol
neg = ret.clip(upper=0)
pos = ret.clip(lower=0)
cands["downside_asym_60"] = (neg.rolling(60).std() / pos.rolling(60).std()).replace([np.inf, -np.inf], np.nan)

# 11. autocorr_20: short-term trend persistence
cands["autocorr_20"] = ret.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False)

# 12. mdd_60: max drawdown depth over 60d
run_max = panel.rolling(60).max()
cands["mdd_60"] = (panel / run_max - 1.0).rolling(60).min()

# 13. wti_copper_cond: commodity-real-activity beta conditional
cands["wti_copper_cond_60x20"] = rolling_beta(ret, ret["WTI"] - ret["COPPER"], 60) * ((panel["WTI"] / panel["WTI"].shift(20) - 1.0) - (panel["COPPER"] / panel["COPPER"].shift(20) - 1.0))

# 14. eurusd_beta_cond_60x20: EUR carry / risk sentiment conditional
cands["eurusd_beta_cond_60x20"] = rolling_beta(ret, eurusd.pct_change(), 60) * (eurusd / eurusd.shift(20) - 1.0)

# 15. hl_amplitude_20: normalized intraday amplitude (high-low)/close
h = pd.concat({s: load_series(s)["high"] for s in TRADABLE}, axis=1).loc[panel.index]
l = pd.concat({s: load_series(s)["low"] for s in TRADABLE}, axis=1).loc[panel.index]
cands["hl_amplitude_20"] = ((h - l) / panel).rolling(20).mean()

print("\n=== SCREEN h=10, gate |IC|>=0.007 & |ICIR|>=0.084 ===")
results = {}
for name, f in cands.items():
    # direction: use sign of mean rank IC (auto-detect by testing +1)
    m = full_validate(f, panel, horizon=10, direction=1, label=name)
    if m:
        results[name] = m

print("\n=== PASS GATE? ===")
for name, m in sorted(results.items(), key=lambda kv: -abs(kv[1]["icir"])):
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    print(f"{name:26s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} "
          f"hit={m['ic_hit_ratio']:.3f} rho={m['max_abs_library_correlation']:.3f} "
          f"{'PASS' if gate else 'fail'}")
