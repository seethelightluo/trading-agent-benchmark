"""miner_1: debug rate_beta_cn10y recent-window NaN + screen new factor ideas."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (
    TRADABLE, load_panels, close_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank,
    decay_profile, max_library_corr, library_signals,
)

panels = load_panels(days=5000)
closes = close_panel(panels)
rets = closes.pct_change()
vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
lib = library_signals(panels, closes, rets, vix)

# ---- debug rate_beta_cn10y ----
cn10 = panels["CN10Y"]["close"].astype(float)
cn10_ret = cn10.pct_change()
beta2 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10_ret.rename("c")], axis=1)
    b = z["a"].rolling(60).cov(z["c"]) / z["c"].rolling(60).var()
    beta2[a] = b
rate_panel = pd.DataFrame(beta2, index=rets.index)
cut = closes.index[-1] - pd.Timedelta(days=3 * 365)
p2 = rate_panel[rate_panel.index >= cut]
print("rate_beta recent window valid counts per asset:")
print(p2.notna().sum().to_string())
print("rate_beta recent window dates with >=8 valid:", int((p2.notna().sum(axis=1) >= 8).sum()))

# ---- candidate screening ----
cands = {}

# C1: USDJPY beta 60d (carry proxy)
jpy = panels["USDJPY"]["close"].astype(float)
jpy_ret = jpy.pct_change()
bj = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), jpy_ret.rename("j")], axis=1)
    b = z["a"].rolling(60).cov(z["j"]) / z["j"].rolling(60).var()
    bj[a] = b
cands["usdjpy_beta_60d"] = pd.DataFrame(bj, index=rets.index)

# C2: DXY beta 60d
dxy = panels["DXY"]["close"].astype(float)
dxy_ret = dxy.pct_change()
bd = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), dxy_ret.rename("d")], axis=1)
    b = z["a"].rolling(60).cov(z["d"]) / z["d"].rolling(60).var()
    bd[a] = b
cands["dxy_beta_60d"] = pd.DataFrame(bd, index=rets.index)

# C3: Trend efficiency 60d: |sum(ret)| / sum(|ret|)
win = 60
num = rets.rolling(win).sum()
den = rets.abs().rolling(win).sum()
cands["trend_efficiency_60d"] = num / den

# C4: Momentum consistency: fraction of positive daily returns over 20d
cands["mom_consistency_20d"] = (rets > 0).rolling(20).mean()

# C5: Short-term reversal 5d (skip1)
cands["reversal_5d_skip1"] = -(closes.shift(1) / closes.shift(6) - 1.0)

# C6: Cross-sectional relative momentum 20d (asset vs median of universe)
mom20 = closes / closes.shift(20) - 1.0
med20 = mom20.median(axis=1)
cands["rel_mom_vs_median_20d"] = mom20.sub(med20, axis=0)

for name, panel in cands.items():
    fwd = forward_returns(closes, 10)
    ics = rank_ic_series(panel, fwd, min_valid=8)
    if len(ics) == 0:
        print(f"\n=== {name}: NO VALID IC DATES ===")
        continue
    m = summarize_ic(ics, 1)
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    # recent 3y
    p2 = panel[panel.index >= cut]
    c2 = closes[closes.index >= cut]
    ics2 = rank_ic_series(p2, forward_returns(c2, 10), 8)
    m2 = summarize_ic(ics2, 1) if len(ics2) else {"ic": None, "icir": None, "n_ic_dates": 0}
    print(f"\n=== {name} FULL ===")
    print(json.dumps({k: m[k] for k in ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "coverage_asset_days",
                                         "coverage_dates_ge8", "turnover_10d_rank", "max_abs_library_correlation",
                                         "max_corr_factor"]}, indent=1))
    print(f"--- recent3y: {json.dumps(m2)}")
