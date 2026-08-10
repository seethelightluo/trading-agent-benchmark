import json, glob, os, numpy as np

print("=== factors/ root (potential active library) ===")
for f in sorted(glob.glob('factors/*.json')):
    if '.bak' in f or 'reason' in f or os.path.isdir(f):
        continue
    try:
        d = json.load(open(f))
        vid = d.get('factor_id')
        status = d.get('validation', {}).get('status')
        ic = d.get('validation', {}).get('metrics', {}).get('ic')
        icir = d.get('validation', {}).get('metrics', {}).get('icir')
        last = d.get('last_validated', d.get('validation', {}).get('timestamp', ''))[:16]
        print(f"{os.path.basename(f):55s} id={vid:35s} st={status} ic={ic} icir={icir} last={last}")
    except Exception as e:
        print(os.path.basename(f), 'ERR', repr(e)[:120])

print()
print("=== npy artifacts in factors/ ===")
for f in sorted(glob.glob('factors/*.npy')):
    try:
        a = np.load(f)
        print(os.path.basename(f), a.shape, 'finite', np.isfinite(a).sum())
    except Exception as e:
        print(os.path.basename(f), 'ERR', repr(e)[:120])
