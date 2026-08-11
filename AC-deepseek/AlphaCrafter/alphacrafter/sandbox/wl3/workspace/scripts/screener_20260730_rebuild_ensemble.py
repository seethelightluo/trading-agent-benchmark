"""Screener 2026-07-30: rebuild factor ensemble with quality-IC tilt + SPX-beta cluster cap.

Method (quality_ic_tilt):
  1. q_i = abs(IC_i) * abs(ICIR_i); direction_i = sign(IC_i)
  2. Crowding exclusion: drop factor if max_abs_library_correlation > 0.7
     with a higher-q partner (library-level, already persisted in each factor file)
  3. Select top-10 by q among remaining
  4. Raw weight w_i = q_i / sum(q_top10)
  5. SPX-beta cluster cap: down_beta_60 + spx_beta_60 <= 0.32.
     If violated, scale cluster to cap and redistribute freed mass
     proportionally to q among non-cluster members (weights still sum to 1).
"""
import json, glob, os, numpy as np, pandas as pd

CLUSTER_CAP = 0.32
CLUSTER = ["down_beta_60", "spx_beta_60"]
MAX_FACTORS = 10

active = [f for f in sorted(glob.glob('factors/*.json'))
          if not f.endswith('.bak') and os.path.basename(f) != 'factor_ensemble.json']

rows = []
for f in active:
    d = json.load(open(f))
    m = d['validation']['metrics']
    rows.append({
        'factor_id': d['factor_id'],
        'ic': m['ic'],
        'icir': m['icir'],
        'hit': m.get('ic_hit_ratio'),
        'q': abs(m['ic']) * abs(m['icir']),
        'dir': 1 if m['ic'] > 0 else -1,
        'max_rho': m.get('max_abs_library_correlation'),
        'max_rho_id': m.get('max_corr_library_id'),
        'turnover_10d_rank': m.get('turnover_10d_rank'),
        'coverage': m.get('coverage_asset_days'),
    })

# 1. crowding exclusion (library correlation > 0.7 with higher-q partner)
excluded = set()
for r in sorted(rows, key=lambda x: -x['q']):
    if r['max_rho'] and r['max_rho'] > 0.7:
        partner = r['max_rho_id']
        pq = next((x['q'] for x in rows if x['factor_id'] == partner), None)
        if pq is not None and pq > r['q']:
            excluded.add(r['factor_id'])

cand = [r for r in sorted(rows, key=lambda x: -x['q']) if r['factor_id'] not in excluded][:MAX_FACTORS]
tot = sum(r['q'] for r in cand)
w = {r['factor_id']: r['q'] / tot for r in cand}

# 2. cluster cap
cs_before = sum(w.get(fid, 0.0) for fid in CLUSTER)
if cs_before > CLUSTER_CAP:
    excess = cs_before - CLUSTER_CAP
    scale = CLUSTER_CAP / cs_before
    for fid in CLUSTER:
        w[fid] *= scale
    non = [r['factor_id'] for r in cand if r['factor_id'] not in CLUSTER]
    ns = sum(w[fid] for fid in non)
    for fid in non:
        w[fid] += excess * (w[fid] / ns)

sel = [{'factor_id': r['factor_id'], 'weight': round(w[r['factor_id']], 10), 'direction': r['dir']}
       for r in cand]

print(f"{'factor_id':<24}{'ic':>8}{'icir':>8}{'hit':>7}{'q':>9}{'dir':>5}{'w':>10}{'turn':>7}")
for r in sorted(cand, key=lambda x: -w[x['factor_id']]):
    print(f"{r['factor_id']:<24}{r['ic']:>8.4f}{r['icir']:>8.4f}{r['hit']:>7.3f}{r['q']:>9.4f}"
          f"{r['dir']:>+5}{w[r['factor_id']]:>10.6f}{r['turnover_10d_rank']:>7.2f}")

tot_w = sum(s['weight'] for s in sel)
cs_after = sum(s['weight'] for s in sel if s['factor_id'] in CLUSTER)
print(f"\nweights sum = {tot_w:.10f}")
print(f"SPX-beta cluster (down_beta+spx_beta) before={cs_before:.4f} after={cs_after:.4f} cap={CLUSTER_CAP}")
print(f"excluded by crowding: {sorted(excluded)}")

out = {
    'schema_version': 1,
    'selected_factors': sel,
    'method': 'quality_ic_tilt',
    'constraints': {
        'spx_beta_cluster_cap': CLUSTER_CAP,
        'cluster_members': CLUSTER,
        'crowding_exclusion_rho': 0.7,
        'max_factors': MAX_FACTORS,
    },
}
with open('factors/factor_ensemble.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print("\nPersisted factors/factor_ensemble.json")

# ---- correlation check on selected signals (last 300 rows, mean CS rank) ----
sig = {}
for fid in [s['factor_id'] for s in sel]:
    arr = np.load(f'factors/{fid}_signal.npy')
    sig[fid] = arr

def cs_rank(a):
    return pd.DataFrame(a).rank(axis=1).values

df = pd.DataFrame({fid: cs_rank(sig[fid][-300:, :]).mean(axis=1) for fid in sig})
corr = df.corr().round(3)
print("\nPairwise corr of mean CS-rank (last 300 rows):")
print(corr.to_string())
mx = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
if len(mx):
    top = mx.sort_values(ascending=False).head(5)
    print("\nTop pairwise corr:")
    for idx, v in top.items():
        print(f"  {idx[0]} ~ {idx[1]} : {v:.3f}")
