"""miner_1 2035-04-16 - re-validate currently effective factors on latest data window.

Factors: vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d (deprecated, check).
Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, ret_panel, forward_returns,
                                 rank_ic_series, summarize_ic, decay_profile,
                                 coverage_metrics, turnover_rank)

t0 = time.time()
panels = load_panels(days=6000)
closes = close_panel(panels)
rets = ret_panel(panels)
print("closes:", closes.shape, closes.index.min().date(), "..", closes.index.max().date(), flush=True)

valid = closes.notna()
print("dates with >=8 valid:", int((valid.sum(axis=1) >= 8).sum()), "of", len(closes), flush=True)

# ---- factor 1: vol_adj_mom_accel_20x60
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
f1 = (mom20 - mom60) / vol20.replace(0, np.nan)

# ---- factor 2: dn_mkt_beta_60d
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_down[a] = z["a"].rolling(60, min_periods=40).cov(z["m"]) / z["m"].rolling(60, min_periods=40).var()
f2 = pd.DataFrame(beta_down, index=rets.index)

# ---- factor 3: rate_beta_cn10y_60d
cn10y_ret = rets["CN10Y"] if "CN10Y" in rets else closes["CN10Y"].pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("c")], axis=1).dropna()
    beta_cn[a] = z["a"].rolling(60, min_periods=40).cov(z["c"]) / z["c"].rolling(60, min_periods=40).var()
f3 = pd.DataFrame(beta_cn, index=rets.index)

fwd10 = forward_returns(closes, 10)

def report(name, fp, expected_sign):
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    print(f"\n=== {name} (dir {expected_sign:+d}) ===", flush=True)
    m = summarize_ic(ics, expected_sign)
    print("FULL:", json_d(m), flush=True)
    for tag, n in [("R125", 125), ("R250", 250), ("R500", 500), ("R1000", 1000)]:
        s = ics[ics.index >= closes.index[-n]]
        if len(s) > 20:
            mm = summarize_ic(s, expected_sign)
            print(f"{tag} n={len(s)}: ic={mm['ic']:.4f} icir={mm['icir']:.4f} hit={mm['ic_hit_ratio']:.2f} std={mm['ic_std']:.3f}", flush=True)
    cov = coverage_metrics(fp, min_valid=8)
    print("coverage:", {k: round(v, 3) for k, v in cov.items()}, flush=True)
    print("turnover_10d_rank:", round(turnover_rank(fp, 10), 3), flush=True)
    dec = decay_profile(fp, closes, (1, 2, 3, 5, 10, 20), 8, expected_sign)
    print("decay:", {k: round(v, 4) for k, v in dec.items()}, flush=True)
    return m

import json as _json
def json_d(x):
    return _json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in x.items()})

report("vol_adj_mom_accel_20x60", f1, 1)
report("dn_mkt_beta_60d", f2, 1)
report("rate_beta_cn10y_60d", f3, -1)
print("\nelapsed_s:", round(time.time() - t0, 1), flush=True)
print("DONE", flush=True)
