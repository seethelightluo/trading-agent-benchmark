"""miner_1 re-validation of existing effective factors + full-period metrics.

Current date 2035-08-20, data through 2035-08-17. Re-validate the 3 persisted
effective factors on the full history and on a recent 3y window.
"""
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

# Recompute the 3 persisted effective factors
sig = {}

# 1) vol_adj_mom_accel_20x60
fast, slow, volw = 20, 60, 20
mom20 = closes.shift(0) / closes.shift(fast) - 1.0
mom60 = closes.shift(0) / closes.shift(slow) - 1.0
vol20 = rets.rolling(volw).std()
sig["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20

# 2) dn_mkt_beta_60d: beta(asset, min(mkt,0), 60)
mkt = rets[TRADABLE].mean(axis=1)
mkt_dn = mkt.where(mkt < 0, 0.0)
beta = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), mkt_dn.rename("m")], axis=1)
    b = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
    beta[a] = b
sig["dn_mkt_beta_60d"] = pd.DataFrame(beta, index=rets.index)

# 3) rate_beta_cn10y_60d: beta(asset, pct_change(CN10Y), 60)
cn10 = panels["CN10Y"]["close"].astype(float)
cn10_ret = cn10.pct_change()
beta2 = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10_ret.rename("c")], axis=1)
    b = z["a"].rolling(60).cov(z["c"]) / z["c"].rolling(60).var()
    beta2[a] = b
sig["rate_beta_cn10y_60d"] = pd.DataFrame(beta2, index=rets.index)

directions = {"vol_adj_mom_accel_20x60": 1, "dn_mkt_beta_60d": 1, "rate_beta_cn10y_60d": -1}

# library signals for correlation check (existing library recompute)
lib = library_signals(panels, closes, rets, vix)

for name, panel in sig.items():
    # full period
    fwd = forward_returns(closes, 10)
    ics = rank_ic_series(panel, fwd, min_valid=8)
    m = summarize_ic(ics, directions[name])
    m.update(coverage_metrics(panel))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, (1, 2, 3, 5, 10, 20), 8, directions[name])
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    # recent 3y window
    cut = closes.index[-1] - pd.Timedelta(days=3 * 365)
    p2 = panel[panel.index >= cut]
    c2 = closes[closes.index >= cut]
    fwd2 = forward_returns(c2, 10)
    ics2 = rank_ic_series(p2, fwd2, min_valid=8)
    m2 = summarize_ic(ics2, directions[name])
    print(f"\n=== {name} (dir={directions[name]}) FULL ===")
    print(json.dumps({k: m[k] for k in ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "coverage_asset_days",
                                         "coverage_dates_ge8", "turnover_10d_rank", "decay_ic_by_horizon",
                                         "max_abs_library_correlation", "max_corr_factor"]}, indent=1))
    print(f"--- {name} RECENT 3Y (>= {cut.date()}) ---")
    print(json.dumps(m2, indent=1))
