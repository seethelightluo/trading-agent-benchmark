"""miner_2 2035-11-30: revalidate the full factor library on fresh panel (data through 2035-11-29)."""
import numpy as np
import pandas as pd
import json, glob, os, sys
sys.path.insert(0, 'scripts')
from miner2_val_lib import load_panel, eval_factor

P = load_panel('scripts/panel_cache_20351129.pkl')
close = P['close']; high = P['high']; low = P['low']; opn = P['open']
vol = P['vol']; ret = P['ret']; macro = P['macro']

def factor_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    if fid.startswith('miner2_20260715_rev_'):
        if fid.endswith('_vs'):
            nd = 1
            s = -(np.log(close) - np.log(close.shift(nd)))
            vmean = vol.rolling(20).mean().replace(0, np.nan)
            vs = vol / vmean
            s = s * np.sign(vs - 1.0)
        else:
            nd = int(fid.split('_')[-1].replace('d', ''))
            s = -(np.log(close) - np.log(close.shift(nd)))
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
        s = np.log(close) - np.log(close.shift(120))
        s = s - (np.log(close) - np.log(close.shift(5)))
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

fids = [os.path.basename(f)[:-5] for f in glob.glob('factors/*.json')
        if '.bak' not in f and 'evicted' not in f and 'quarantine' not in f and 'rejected' not in f]
rows = []
print("=== FACTOR LIBRARY REVALIDATION (data through 2035-11-29) ===")
for fid in sorted(fids):
    direction = -1 if 'vix_beta' in fid else 1
    sig = factor_signal(fid).replace([np.inf, -np.inf], np.nan) * direction
    full = eval_factor(sig, close, horizons=(1, 2, 3, 5, 10), min_n=8)
    w1y = eval_factor(sig, close, horizons=(1, 5, 10), min_n=8, start=sig.index[-252])
    w6m = eval_factor(sig, close, horizons=(1, 5, 10), min_n=8, start=sig.index[-126])
    w3m = eval_factor(sig, close, horizons=(1, 5, 10), min_n=8, start=sig.index[-63])
    row = {'factor_id': fid, 'direction': direction}
    for k in ['ic', 'icir', 'hit', 'n']:
        row[f'1_{k}'] = full[1][k]
    row['ic5'] = full[5]['ic']; row['ic10'] = full[10]['ic']
    row['icir10'] = full[10]['icir']
    row['coverage'] = full['coverage']; row['turnover'] = full['turnover_1d_rank']
    row['n_dates'] = full['n_dates']
    row['ic1_1y'] = w1y[1]['ic']; row['icir1_1y'] = w1y[1]['icir']
    row['ic1_6m'] = w6m[1]['ic']; row['icir1_6m'] = w6m[1]['icir']
    row['ic1_3m'] = w3m[1]['ic']; row['icir1_3m'] = w3m[1]['icir']
    row['ic10_1y'] = w1y[10]['ic']; row['ic10_6m'] = w6m[10]['ic']
    rows.append(row)
    print(f"{fid} dir={direction:+d} | IC1={row['1_ic']:.4f} ICIR1={row['1_icir']:.3f} hit1={row['1_hit']:.3f} n1={int(row['1_n'])} | "
          f"1y {row['ic1_1y']:.4f}/{row['icir1_1y']:.3f} 6m {row['ic1_6m']:.4f}/{row['icir1_6m']:.3f} 3m {row['ic1_3m']:.4f}/{row['icir1_3m']:.3f} | "
          f"IC5={row['ic5']:.4f} IC10={row['ic10']:.4f}/{row['icir10']:.3f} | cov={row['coverage']:.3f} turn={row['turnover']:.3f}")

df = pd.DataFrame(rows)
df.to_csv('scripts/miner2_reval_20351129_lib.csv', index=False)
print("\nsaved scripts/miner2_reval_20351129_lib.csv")
