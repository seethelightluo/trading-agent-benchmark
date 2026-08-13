"""miner_1 (2034-06-12): re-validation of the 3 currently EFFECTIVE library factors
through the latest visible trading day (2034-06-09).

Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, library_signals,
                                 max_library_corr)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

log("loading panels...")
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt = rets.mean(axis=1)
log(f"closes {closes.shape} {closes.index.min().date()} -> {closes.index.max().date()}")

# ---- factor 1: vol_adj_mom_accel_20x60 ----
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std(ddof=0)
f1 = (mom20 - mom60) / vol20

# ---- factor 2: dn_mkt_beta_60d ----
dn = np.minimum(mkt, 0.0)
f2 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                       / dn.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)

# ---- factor 3: rate_beta_cn10y_60d ----
cn_ret = closes["CN10Y"].pct_change()
f3 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                       / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)

factors = {
    "vol_adj_mom_accel_20x60": f1,
    "dn_mkt_beta_60d": f2,
    "rate_beta_cn10y_60d": f3,
}

fwd10 = forward_returns(closes, 10)
ADM = {"ic": 0.0070, "icir": 0.0840}

print("=" * 72)
print("FULL-HISTORY REVALIDATION 2020..2034-06-09 (h=10)")
print("=" * 72)
for name, fp in factors.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    full = summarize_ic(ics, expected_sign=1)
    dec = decay_profile(fp, closes)
    print(f"FACTOR {name}")
    print(" FULL:", json.dumps(full))
    print(" decay:", {k: round(v, 4) for k, v in dec.items()})

print("=" * 72)
print("RECENT WINDOW REVALIDATION drift check")
print("=" * 72)
for name, fp in factors.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    rec3y = summarize_ic(ics[ics.index >= pd.Timestamp("2031-01-01")], expected_sign=1)
    rec1y = summarize_ic(ics[ics.index >= pd.Timestamp("2033-06-01")], expected_sign=1)
    print(f"{name}: recent3y {json.dumps(rec3y)} | recent1y {json.dumps(rec1y)}")

log("done")
