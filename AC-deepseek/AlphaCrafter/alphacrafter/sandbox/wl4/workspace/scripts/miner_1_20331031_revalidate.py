"""miner_1 (2033-10-31): re-validation of the 3 currently EFFECTIVE library factors.

Data through the previous completed trading day (sim clock 2033-10-31 -> visible
through 2033-10-28). Full window 2020..2033-10 + recent 2y window for drift check.
Admission gate (h=10): |IC| >= 0.0070 and |ICIR| >= 0.0840.

Factor defs (from factors/*.json):
  vol_adj_mom_accel_20x60 : (mom20 - mom60) / vol20          (dir +1)
  dn_mkt_beta_60d         : beta(asset, min(mkt,0), 60)      (dir +1, low-beta favored)
  rate_beta_cn10y_60d     : beta(asset, dCN10Y, 60)          (dir -1, low-beta favored)
"""
import sys, json, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

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
recent_cut = pd.Timestamp("2031-10-28")
ADM = {"ic": 0.0070, "icir": 0.0840}

results = {}
for name, fp in factors.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    full = summarize_ic(ics, expected_sign=1)
    rec = summarize_ic(ics[ics.index >= recent_cut], expected_sign=1)
    dec = decay_profile(fp, closes)
    to = turnover_rank(fp, 10)
    cov = coverage_metrics(fp)
    full_pass = abs(full["ic"]) >= ADM["ic"] and abs(full["icir"]) >= ADM["icir"]
    rec_pass = abs(rec["ic"]) >= ADM["ic"] and abs(rec["icir"]) >= ADM["icir"]
    results[name] = {"full": full, "recent2y": rec, "decay": dec,
                     "turnover_10d_rank": to, "coverage": cov,
                     "full_pass": full_pass, "recent2y_pass": rec_pass}
    print("=" * 72)
    print(f"FACTOR {name}")
    print(" FULL :", json.dumps(full))
    print(" RECENT2y:", json.dumps(rec))
    print(" decay(h1..20):", {k: round(v, 4) for k, v in dec.items()})
    print(" turnover_10d_rank:", to, "| coverage:", json.dumps(cov))
    print(f" gate: full={full_pass} (|IC|>={ADM['ic']},|ICIR|>={ADM['icir']})  recent2y={rec_pass}")

# pairwise IC-series correlation (provenance/audit)
ics_map = {n: rank_ic_series(factors[n].replace([np.inf, -np.inf], np.nan), fwd10) for n in factors}
base = pd.concat(ics_map, axis=1)
print("\n-- pairwise IC-series corr --")
print(base.corr().round(3).to_string())
log("DONE")
