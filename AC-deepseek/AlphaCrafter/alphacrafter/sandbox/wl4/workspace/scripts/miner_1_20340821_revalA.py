"""miner_1 (2034-08-21) PART A: re-validate the 3 currently EFFECTIVE library
factors through the latest visible trading day. Instrumented + fast."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

log("loading panels...")
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt = rets.mean(axis=1)
log(f"closes {closes.shape} {closes.index.min().date()} -> {closes.index.max().date()}")

mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std(ddof=0)
f1 = (mom20 - mom60) / vol20

dn = np.minimum(mkt, 0.0)
f2 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(dn)
                       / dn.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)
cn_ret = closes["CN10Y"].pct_change()
f3 = pd.DataFrame({a: (rets[a].rolling(60, min_periods=40).cov(cn_ret)
                       / cn_ret.rolling(60, min_periods=40).var()) for a in rets.columns},
                  index=rets.index)

lib_factors = {
    "vol_adj_mom_accel_20x60": f1,
    "dn_mkt_beta_60d": f2,
    "rate_beta_cn10y_60d": f3,
}
fwd10 = forward_returns(closes, 10)
ADM_IC, ADM_ICIR = 0.0070, 0.0840

log("computing rank IC...")
print("=" * 100)
print("PART A: LIBRARY RE-VALIDATION (h=10) through", closes.index.max().date())
print("=" * 100)
reval = {}
for name, fp in lib_factors.items():
    fp = fp.replace([np.inf, -np.inf], np.nan)
    ics = rank_ic_series(fp, fwd10, min_valid=8)
    full = summarize_ic(ics, expected_sign=1)
    rec = {}
    for label, start in [("since2022", "2022-01-01"), ("since2024", "2024-01-01"),
                         ("since2024-08", "2024-08-01"), ("last500d", None)]:
        sub = ics[ics.index >= pd.Timestamp(start)] if start else ics.iloc[-500:]
        rec[label] = summarize_ic(sub, expected_sign=1)
    dec = decay_profile(fp, closes)
    cov = coverage_metrics(fp, min_valid=8)
    turn = turnover_rank(fp, 10)
    p_full = abs(full["ic"]) >= ADM_IC and abs(full["icir"]) >= ADM_ICIR
    p_rec = any(abs(v["ic"]) >= ADM_IC and abs(v["icir"]) >= ADM_ICIR for v in rec.values())
    reval[name] = {"full": full, "recent": rec, "decay": dec, "cov": cov,
                   "turn": turn, "pass_full": p_full, "pass_recent": p_rec}
    print(f"FACTOR {name}")
    print(f"  FULL: IC={full['ic']:+.4f} ICIR={full['icir']:+.3f} hit={full['ic_hit_ratio']:.2f} n={full['n_ic_dates']} -> {'PASS' if p_full else 'FAIL'}")
    for k, v in rec.items():
        ok = abs(v["ic"]) >= ADM_IC and abs(v["icir"]) >= ADM_ICIR
        print(f"  {k:12s}: IC={v['ic']:+.4f} ICIR={v['icir']:+.3f} hit={v['ic_hit_ratio']:.2f} n={v['n_ic_dates']:5d} -> {'PASS' if ok else 'fail'}")
    print(f"  decay: { {k: round(v,4) for k,v in dec.items()} }  covA={cov['coverage_asset_days']} covD={cov['coverage_dates_ge8']} turn={turn}")

json.dump({k: {kk: (vv if not isinstance(vv, dict) else {x: (y if not isinstance(y, dict) else y) for x, y in vv.items()}) for kk, vv in v.items()} for k, v in reval.items()},
          open("scripts/_miner1_20340821_revalA.json", "w"), indent=1, default=str)
log("done -> scripts/_miner1_20340821_revalA.json")
