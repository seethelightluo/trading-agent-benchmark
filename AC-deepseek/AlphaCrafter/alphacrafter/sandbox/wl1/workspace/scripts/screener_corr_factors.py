import json, glob, os, gzip, base64
import numpy as np

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if not f.endswith('.bak')]
seen = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    if d['factor_id'] not in seen:
        seen[d['factor_id']] = d

def load_signal(d):
    sa = d.get('signal_artifact')
    if isinstance(sa, dict) and 'data_b64' in sa:
        raw = base64.b64decode(sa['data_b64'])
        arr = np.frombuffer(gzip.decompress(raw), dtype=np.float32).reshape(sa['n_dates'], sa['n_symbols'])
        return arr
    elif isinstance(sa, str) and sa.endswith('.npy'):
        return np.load(os.path.join('factors', sa))
    return None

sigs = {}
for fid, d in seen.items():
    arr = load_signal(d)
    if arr is not None:
        sigs[fid] = arr

fids = list(sigs.keys())
print('loaded signals:', len(fids))
for fid, arr in sigs.items():
    print(' ', fid, arr.shape, 'valid frac', round(float((~np.isnan(arr)).mean()), 3))

n = min(arr.shape[0] for arr in sigs.values())
print('common rows:', n)

def datewise_corr(a, b):
    a, b = a[:n], b[:n]
    cs = []
    for i in range(n):
        va, vb = a[i], b[i]
        ok = ~(np.isnan(va) | np.isnan(vb))
        if ok.sum() >= 5:
            ra = np.argsort(np.argsort(va[ok])).astype(float)
            rb = np.argsort(np.argsort(vb[ok])).astype(float)
            ra = ra - ra.mean(); rb = rb - rb.mean()
            denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
            if denom > 0:
                cs.append((ra * rb).sum() / denom)
    return np.nanmean(cs) if cs else np.nan

names = fids
K = len(names)
M = np.zeros((K, K))
for i in range(K):
    for j in range(i + 1, K):
        c = datewise_corr(sigs[names[i]], sigs[names[j]])
        M[i, j] = M[j, i] = c

np.set_printoptions(precision=2, suppress=True, linewidth=250)
print('Date-wise avg cross-sectional Spearman corr matrix:')
hdr = ' '.join(f'{x[:10]:>10s}' for x in names)
print(' ' * 24, hdr)
for i in range(K):
    print(f'{names[i][:22]:>22s}', ' '.join(f'{M[i, j]:>10.2f}' for j in range(K)))

np.save('scripts/_factor_corr_matrix.npy', M)
with open('scripts/_factor_corr_names.txt', 'w') as fh:
    fh.write('\n'.join(names))
print('saved.')
