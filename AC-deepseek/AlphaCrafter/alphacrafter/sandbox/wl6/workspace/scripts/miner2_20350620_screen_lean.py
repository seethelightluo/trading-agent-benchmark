"""miner_2 lean screen (2035-06-20). Lightweight IC evaluation at horizon 10.
Only data visible through 2035-06-19. Reports n dates/instruments.
Per-candidate: compute signal, fwd returns, daily rank IC series -> mean IC, ICIR.
Decay + lib_corr only for candidates passing the gate.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, rank_ic_series, align_fwd_returns
import pandas as pd, numpy as np
from scipy.stats import rankdata

VIS = "2035-06-19"
px = load_panel(max_date=VIS)
print("panel:", px.shape, "n_dates", px.shape[0], "n_assets", px.shape[1], flush=True)
ret = px.pct_change()
vix = load_macro("VIX", max_date=VIS)
fwd10 = align_fwd_returns(px, 10)

def eval_ic(f):
    ic = rank_ic_series(f, fwd10)
    if len(ic) < 5:
        return None
    m = ic.mean(); s = ic.std(ddof=1)
    icir = m / s if s > 0 else float('nan')
    return {"ic": round(float(m), 4), "icir": round(float(icir), 4),
            "n_dates": int(len(ic)), "hit": round(float((ic > 0).mean()), 3)}

cands = {}
mom5 = px / px.shift(5) - 1
mom10 = px / px.shift(10) - 1
mom20 = px / px.shift(20) - 1

cands["mom_20d_skip5"] = mom20.rank(axis=1, pct=True)
vix_lvl = vix.reindex(px.index, method='ffill')
vix_ratio = vix_lvl / vix_lvl.rolling(60).mean()
cands["vix_fall_catchup5"] = (-mom5.rank(axis=1, pct=True)) * (1.0 - vix_ratio.clip(0, 2))
cands["vol_adj_mom5"] = ret.rolling(5).sum() / ret.rolling(10).std()
up = ret.clip(lower=0).rolling(20).std()
dn = ret.clip(upper=0).rolling(20).std()
tot = ret.rolling(60).std()
cands["updown_vol_imbalance20"] = (up - dn) / tot
pos60 = (ret > 0).rolling(60).mean()
cands["consistency60_cs"] = pos60 - pos60.mean(axis=1)

passed = []
for name, f in cands.items():
    r = eval_ic(f)
    if r is None:
        print(f"[{name}] insufficient", flush=True); continue
    gate = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    print(f"[{name}] IC={r['ic']} ICIR={r['icir']} n={r['n_dates']} hit={r['hit']} gate={'PASS' if gate else 'FAIL'}", flush=True)
    if gate:
        passed.append((name, f, r))

print("\n===== GATE-PASSING CANDIDATES: depth (decay + lib_corr) =====", flush=True)
for name, f, r in passed:
    # decay
    decay = {}
    for h in (3, 5, 10, 20):
        ic_h = rank_ic_series(f, align_fwd_returns(px, h))
        decay[str(h)] = round(float(ic_h.mean()), 4) if len(ic_h) else None
    r["decay"] = decay
    print(f"{name}: IC={r['ic']} ICIR={r['icir']} n={r['n_dates']} hit={r['hit']} decay={decay}", flush=True)

if not passed:
    print("No candidate passed the gate.", flush=True)