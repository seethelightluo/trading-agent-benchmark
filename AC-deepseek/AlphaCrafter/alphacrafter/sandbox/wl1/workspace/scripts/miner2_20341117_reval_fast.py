"""miner_2 2034-11-03: revalidate the factor library on fresh panel (data through 2034-11-16). Optimized numpy rank IC."""
import numpy as np
import pandas as pd
import json, glob, os, time

t0 = time.time()
with open('scripts/panel_cache_20341117.pkl', 'rb') as f:
    P = pd.read_pickle(f)
close = P['close']; high = P['high']; low = P['low']; opn = P['open']; vol = P['vol']; ret = P['ret']; macro = P['macro']

def ln(x):
    return np.log(x)

def factor_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    if fid.startswith('miner2_20260715_rev_'):
        if fid.endswith('_vs'):
            nd = 1
            s = -(ln(close) - ln(close.shift(nd)))
            vmean = vol.rolling(20).mean().replace(0, np.nan)
            vs = vol / vmean
            s = s * np.sign(vs - 1.0)
        else:
            nd = int(fid.split('_')[-1].replace('d', ''))
            s = -(ln(close) - ln(close.shift(nd)))
        return s
    if fid.startswith('miner2_20260715_nclv_'):
        nd = int(fid.split('_')[-1].replace('d', ''))
        rnd = ret.rolling(nd).mean()
        vnd = ret.rolling(nd).std()
        s = -(rnd / vnd.replace(0, np.nan))
        return s
    if fid.startswith('miner2_20260715_nbody_'):
        body = (close - opn) / opn
        rng = (high - low) / close
        s = -(body / rng.replace(0, np.nan))
        return s
    if fid.startswith('miner2_20260715_id_rev_'):
        gap = opn / close.shift(1) - 1.0
        s = -gap
        return s
    if fid == 'mom_120d_skip5':
        s = ln(close) - ln(close.shift(120))
        s = s - (ln(close) - ln(close.shift(5)))
        return s
    if fid == 'vol_of_vol20x60':
        v20 = ret.rolling(20).std()
        v60 = ret.rolling(60).std()
        s = v20 / v60 - 1.0
        return s
    if fid == 'vix_beta_cond_60x20':
        vix = macro['VIX']
        dvix = vix.pct_change()
        s = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for c in close.columns:
            r = ret[c]
            rb = r.rolling(60).corr(dvix) * (r.rolling(60).std() / dvix.rolling(60).std())
            s[c] = rb
        return s
    raise ValueError('unknown factor ' + fid)

def rank_ic_fast(sig, fwd):
    fr = ret.shift(-fwd)
    dates = sig.index
    S = sig.values.astype(float)
    F = fr.values.astype(float)
    out_dates, out_ics = [], []
    for i in range(len(dates)):
        s = S[i]; f = F[i]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() >= 8:
            ss = s[m]; ff = f[m]
            sr = ss.argsort().argsort().astype(float)
            frr = ff.argsort().argsort().astype(float)
            ic = float(np.corrcoef(sr, frr)[0, 1])
            if np.isfinite(ic):
                out_dates.append(dates[i]); out_ics.append(ic)
    return pd.Series(out_ics, index=pd.DatetimeIndex(out_dates))

def report(fid, direction):
    sig = factor_signal(fid)
    sig = sig.replace([np.inf, -np.inf], np.nan)
    sig = sig * direction
    out = {'factor_id': fid, 'direction': direction}
    for h in [1, 2, 3, 5, 10]:
        ics = rank_ic_fast(sig, h)
        if len(ics) == 0:
            out[f'ic{h}'] = np.nan; out[f'icir{h}'] = np.nan; out[f'n{h}'] = 0
            continue
        ic = float(ics.mean())
        icir = float(ics.mean() / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else np.nan
        out[f'ic{h}'] = ic; out[f'icir{h}'] = icir; out[f'n{h}'] = len(ics)
        for label, wd in [('2y', 504), ('1y', 252), ('6m', 126)]:
            sub = ics.tail(wd)
            if len(sub) >= 60:
                out[f'ic{h}_{label}'] = float(sub.mean())
                out[f'icir{h}_{label}'] = float(sub.mean() / sub.std() * np.sqrt(len(sub))) if sub.std() > 0 else np.nan
            else:
                out[f'ic{h}_{label}'] = np.nan; out[f'icir{h}_{label}'] = np.nan
    return out

fids = sorted([os.path.basename(f)[:-5] for f in glob.glob('factors/*.json')
               if '.bak' not in f and os.path.dirname(f) == 'factors'])
rows = []
print("=== FACTOR LIBRARY REVALIDATION (data through 2034-11-16, VIX 100.6 crisis) ===", flush=True)
for fid in fids:
    try:
        d = json.load(open(f'factors/{fid}.json'))
    except Exception as e:
        print('skip', fid, e); continue
    direction = -1 if 'vix_beta' in fid else 1
    r = report(fid, direction)
    rows.append(r)
    print(f"{fid} dir={direction:+d} | ic1 {r['ic1']:.4f}/{r['icir1']:.3f} (n{r['n1']}) 1y {r.get('ic1_1y', np.nan):.4f}/{r.get('icir1_1y', np.nan):.3f} 6m {r.get('ic1_6m', np.nan):.4f}/{r.get('icir1_6m', np.nan):.3f} | ic5 {r['ic5']:.4f}/{r['icir5']:.3f} | ic10 {r['ic10']:.4f}/{r['icir10']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv('scripts/miner2_reval_20341103_lib.csv', index=False)
print('\nsaved scripts/miner2_reval_20341103_lib.csv', flush=True)

print("\n=== GATE CHECK (|ic|>=0.007 & |icir|>=0.084) ===", flush=True)
for _, row in df.iterrows():
    fid = row['factor_id']
    passed = []
    for h in [1, 2, 3, 5, 10]:
        ic = row[f'ic{h}']; icir = row[f'icir{h}']
        if np.isfinite(ic) and np.isfinite(icir) and abs(ic) >= 0.007 and abs(icir) >= 0.084:
            passed.append(f"h{h}:{ic:+.4f}/{icir:+.3f}")
    print(f"{fid}: {'PASS ' + ' '.join(passed) if passed else 'FAIL'}", flush=True)
print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
