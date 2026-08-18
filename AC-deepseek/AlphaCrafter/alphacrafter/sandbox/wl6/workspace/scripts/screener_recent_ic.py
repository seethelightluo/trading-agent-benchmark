"""Screener: compute recent cross-sectional IC / ICIR for active factors up to the visible date."""
import json, base64, zlib, io, glob
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

VISIBLE = '2029-05-16'
ASSETS = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

px = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VISIBLE].set_index('date')['close']
    px[a] = df
px = pd.DataFrame(px).sort_index()
fwd10 = px.shift(-10)/px - 1.0

def load_factor(fp):
    with open(fp) as f:
        d = json.load(f)
    art = d['validation']['signal_artifact']
    df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art['data'])).decode()))
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')[ASSETS]
    return d['factor_id'], df[df.index <= VISIBLE]

factors = {}
for fp in sorted(glob.glob('factors/*.json')):
    if '.bak' in fp or 'ensemble' in fp: continue
    try:
        fid, panel = load_factor(fp)
        factors[fid] = panel
    except Exception as e:
        print('skip', fp, e)

results = []
for fid, panel in factors.items():
    dates = panel.index.intersection(fwd10.index)
    ic_list = []
    for dt in dates[-150:]:
        x = panel.loc[dt]; y = fwd10.loc[dt]
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 8:
            try:
                ic, _ = spearmanr(x[mask], y[mask])
            except Exception:
                continue
            if np.isfinite(ic): ic_list.append(ic)
    if len(ic_list) >= 20:
        ic_arr = np.array(ic_list)
        results.append((fid, ic_arr.mean(), ic_arr.mean()/ic_arr.std() if ic_arr.std()>0 else 0.0, len(ic_arr), ic_arr[-25:].mean()))

results.sort(key=lambda r: -abs(r[1]))
print(f'Recent IC (Spearman, fwd 10d) up to {VISIBLE}')
print(f'{"factor":24s} {"IC_150d":>8s} {"ICIR":>7s} {"n":>4s} {"IC_last25":>10s}')
for fid, ic, icir, n, ic25 in results:
    print(f'{fid:24s} {ic:8.4f} {icir:7.3f} {n:4d} {ic25:10.4f}')

fids = [r[0] for r in results[:12]]
all_dates = sorted(set().union(*[set(factors[f].index) for f in fids]))
all_dates = [d for d in all_dates if d <= pd.Timestamp(VISIBLE)][-60:]
stack = []
for dt in all_dates:
    vec = []
    for fid in fids:
        p = factors[fid]
        if dt in p.index:
            s = p.loc[dt]
            vec.append(s.rank(pct=True).values)
        else:
            vec.append(np.full(len(ASSETS), np.nan))
    stack.append(np.concatenate([v[None,:] for v in vec], axis=0))
stack = np.vstack(stack)
C = np.corrcoef(stack.T)
print()
print('Pairwise correlation (recent 60d, cross-sectional ranks):')
hdr = ' '.join(f'{n[:6]:>7s}' for n in fids)
print(f'{"":8s} {hdr}')
for i in range(len(fids)):
    row = ' '.join(f'{C[i,j]:7.2f}' for j in range(len(fids)))
    print(f'{fids[i][:8]:8s} {row}')
