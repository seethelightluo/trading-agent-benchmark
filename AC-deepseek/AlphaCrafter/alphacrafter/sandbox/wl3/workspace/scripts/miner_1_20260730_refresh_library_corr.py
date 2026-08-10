"""miner_1 2026-07-30: recompute pairwise library correlations among remaining
effective factors from real signal artifacts; refresh max_abs_library_correlation
metadata; report any remaining pair >= 0.5 for further pruning.
"""
import json, glob, os
import numpy as np

def load_signal(d):
    art = d.get('signal_artifact')
    if not art:
        return None
    p = f'factors/{art}'
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)

rows = []
for p in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in p or 'deprecated' in p:
        continue
    d = json.load(open(p))
    if d['validation']['status'] != 'EFFECTIVE':
        continue
    s = load_signal(d)
    if s is None:
        print(f'  WARN: {d["factor_id"]} has no loadable signal artifact', flush=True)
        continue
    rows.append((d['factor_id'], d, s))

n = len(rows)
print(f'=== recompute pairwise rho among {n} effective factors ===', flush=True)
X = np.column_stack([s for _, _, s in rows])
ids = [r[0] for r in rows]
# drop rows (dates) with any NaN across all signals for a clean cross-sectional estimate
mask = np.isfinite(X).all(axis=1)
Xm = X[mask]
print(f'  usable dates: {mask.sum()}/{len(mask)}', flush=True)
R = np.corrcoef(Xm.T)
np.fill_diagonal(R, 0.0)

hot = []
for i in range(n):
    for j in range(i+1, n):
        r = abs(R[i, j])
        if r >= 0.5:
            hot.append((ids[i], ids[j], R[i, j]))

print(f'\n=== remaining pairs with |rho| >= 0.5: {len(hot)} ===', flush=True)
for a, b, r in sorted(hot, key=lambda t: -abs(t[2])):
    print(f'  {a:24s} vs {b:24s} rho={r:+.3f}', flush=True)

print('\n=== refreshed per-factor max library correlation ===', flush=True)
for i, (fid, d, s) in enumerate(rows):
    r = np.abs(R[i])
    j = int(np.argmax(r))
    rho = float(r[j])
    d['validation']['metrics']['max_abs_library_correlation'] = round(rho, 4)
    d['validation']['metrics']['max_corr_library_id'] = ids[j]
    d['validation']['metrics']['library_pairwise_n'] = n
    d['validation']['last_validated'] = '2026-07-30'
    json.dump(d, open(f'factors/{fid}.json', 'w'), indent=2, default=str)
    print(f'  {fid:24s} rho={rho:+.4f} vs {ids[j]}', flush=True)

print('\nDONE')
