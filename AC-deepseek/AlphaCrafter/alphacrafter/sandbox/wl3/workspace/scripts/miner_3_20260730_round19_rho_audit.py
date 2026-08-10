"""Audit: recompute pairwise Spearman rho between newly persisted factors and the
full library using the ACTUAL saved artifacts on the current canonical grid —
this mirrors what the deterministic gate will do."""
import numpy as np, pandas as pd, json, glob

pairs = [('er_20', 'vol_adj_mom_20_60'),
         ('rev_250_20', 'mom_accel_60_120'),
         ('er_20', 'efficiency_ratio_20')]

def load_rank(fid):
    # find artifact via json if exists else direct npy
    try:
        d = json.load(open(f'factors/{fid}.json'))
        art = d.get('signal_artifact')
        p = f'factors/{fid}.json' if not art else f'factors/{art}'
    except Exception:
        p = f'factors/{fid}_signal.npy'
    arr = np.load(p, allow_pickle=False)
    out = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        v = np.isfinite(row)
        if v.sum() >= 3:
            out[i, v] = pd.Series(row[v]).rank().values
    return arr, out

def mean_cs_pearson_rho(a, b):
    """mean of daily cross-sectional Pearson rho on ranks (matches gate style)."""
    corrs = []
    for t in range(min(len(a), len(b))):
        x = a[t]; y = b[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            xv = x[m]; yv = y[m]
            xc = xv - xv.mean(); yc = yv - yv.mean()
            den = np.sqrt((xc**2).sum() * (yc**2).sum())
            if den > 0:
                corrs.append((xc * yc).sum() / den)
    return float(np.mean(corrs)), len(corrs)

for fa, fb in pairs:
    try:
        arra, ranka = load_rank(fa)
        arrb, rankb = load_rank(fb)
        print(f"{fa} shape={arra.shape} | {fb} shape={arrb.shape}")
        r, n = mean_cs_pearson_rho(ranka, rankb)
        print(f"  mean daily cs rho = {r:.4f} (n={n} dates) -> {'CONFLICT >=0.5' if abs(r) >= 0.5 else 'OK <0.5'}")
    except Exception as e:
        print(f"{fa} vs {fb}: ERR {e}")

# also er_20 vs everything in the effective library (full scan)
libs = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        _, rk = load_rank(fid)
        libs[fid] = rk
    except Exception:
        pass
print("\nFull scan for er_20 / rev_250_20 against effective library:")
for newf in ['er_20', 'rev_250_20']:
    _, rk_new = load_rank(newf)
    rows = []
    for fid, rk in libs.items():
        if fid == newf:
            continue
        try:
            r, n = mean_cs_pearson_rho(rk_new, rk)
            rows.append((abs(r), fid, r))
        except Exception:
            pass
    rows.sort(reverse=True)
    print(f"  {newf}: " + "; ".join(f"{fid}={r:.3f}" for _, fid, r in rows[:5]))
