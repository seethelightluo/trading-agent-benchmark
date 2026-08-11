"""miner_1 cycle 2026-07-30: withdraw dd_pos_60 (redundant with semi_down_ratio_20).

dd_pos_60 passed the IC/ICIR gate (IC=0.0323, ICIR=0.0964) but a full-library artifact
correlation check shows it is a near mirror image of the already-active factor
semi_down_ratio_20 (abs Spearman rho > 0.6, above the 0.5 correlation threshold), and its
quality (ic*icir ~ 0.0031) is far below semi_down_ratio_20's (~0.031). It would be evicted
by the deterministic gate as a pairwise-correlation conflict with lower quality; we move it
to evicted/ proactively with full provenance.
"""
import json, os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from factor_validate import closes_panel
from miner3_lib import decode_artifact, LIB_FACTORS

VIS = '2026-07-29'
close = closes_panel(VIS)
rmax60 = close.rolling(60, min_periods=36).max()
rmin60 = close.rolling(60, min_periods=36).min()
sig_dd = (close - rmin60) / (rmax60 - rmin60)

# spearman rho vs library artifacts (flattened), matching gate's abs_spearman_rho
conflicts = []
for fid in LIB_FACTORS:
    p = f'factors/{fid}.json'
    if not os.path.exists(p) or fid == 'dd_pos_60':
        continue
    d = json.load(open(p))
    art = d.get('validation', {}).get('signal_artifact')
    if not art:
        continue
    libp = decode_artifact(art)
    common = sig_dd.index.intersection(libp.index)
    a = sig_dd.loc[common].stack()
    b = libp.loc[common].stack()
    m = a.notna() & b.notna()
    if m.sum() >= 200:
        r = float(a[m].rank().corr(b[m].rank()))
        if np.isfinite(r):
            conflicts.append({"factor_id": fid, "abs_spearman_rho": round(abs(r), 4)})

conflicts.sort(key=lambda x: -x["abs_spearman_rho"])
print("conflicts (sorted):", json.dumps(conflicts, indent=1))

# quality of dd_pos_60 and of the conflicting incumbent
ic = 0.0323
icir = 0.0964
quality = round(ic * icir, 8)
inc = json.load(open('factors/semi_down_ratio_20.json'))
inc_sel = inc.get('benchmark_admission', {}).get('selected_metrics', {})
inc_q = inc_sel.get('quality', inc['validation']['metrics']['ic'] * (inc['validation']['metrics']['icir'] or 0))
print(f"dd_pos_60 quality={quality}; semi_down_ratio_20 quality={round(inc_q, 8)}")

reason = {
    "source": "dd_pos_60.json",
    "factor_id": "dd_pos_60",
    "reason": "pairwise correlation conflict; lower quality",
    "quality": quality,
    "conflicts": conflicts[:3],
    "contract": {
        "ic_threshold": 0.007,
        "icir_threshold": 0.084,
        "correlation_threshold": 0.5,
        "library_capacity": 30,
        "active_top_k": 10
    },
    "note": "miner_1 proactive eviction 2026-07-30: passes IC/ICIR gate but abs Spearman rho vs "
            "semi_down_ratio_20 exceeds 0.5; predictive content is a mirror of the incumbent."
}

src = 'factors/dd_pos_60.json'
assert os.path.exists(src), "dd_pos_60.json not found"
os.replace(src, 'factors/evicted/dd_pos_60.json')
with open('factors/evicted/dd_pos_60.json.reason.json', 'w') as f:
    json.dump(reason, f, indent=1)
print("EVICTED dd_pos_60 -> factors/evicted/dd_pos_60.json (+reason)")
print("active library now:", sorted(f for f in os.listdir('factors') if f.endswith('.json') and f != 'factor_ensemble.json'))
