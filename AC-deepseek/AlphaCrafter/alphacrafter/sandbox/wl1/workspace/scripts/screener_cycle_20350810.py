"""Screener 2035-08-10 cycle: regime assessment + fresh IC panel through 2035-08-09."""
import pandas as pd, numpy as np, json, time, os
t0 = time.time()

symbols = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
asof = '2035-08-09'

dfs = {}
for s in symbols:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= asof].set_index('date').sort_index()
    dfs[s] = df[['open','high','low','close','volume']]

px = pd.DataFrame({s: dfs[s]['close'] for s in symbols})
op = pd.DataFrame({s: dfs[s]['open'] for s in symbols})
hi = pd.DataFrame({s: dfs[s]['high'] for s in symbols})
lo = pd.DataFrame({s: dfs[s]['low'] for s in symbols})
vo = pd.DataFrame({s: dfs[s]['volume'] for s in symbols})
print('panel shape', px.shape, 'last', px.index[-1].date(), 't=%.1fs' % (time.time()-t0))

ret = px.pct_change()
lpx = np.log(px)

# ---------- REGIME ----------
print('\n=== ASSET PANEL (through %s) ===' % asof)
rows = []
for a in symbols:
    s = px[a]; r = ret[a]
    last = s.iloc[-1]
    def cum(n): return s.iloc[-1]/s.iloc[-1-n]-1 if len(s)>n else np.nan
    ma20 = s.rolling(20).mean().iloc[-1]
    ma60 = s.rolling(60).mean().iloc[-1]
    rows.append(dict(asset=a, last=last, r20=cum(20), r60=cum(60), r120=cum(120),
                     above_ma20=1 if last>ma20 else 0, above_ma60=1 if last>ma60 else 0,
                     vol20=r.tail(20).std()*np.sqrt(252), vol60=r.tail(60).std()*np.sqrt(252)))
tab = pd.DataFrame(rows).set_index('asset')
print(tab.round(4).to_string())
print('\nEqW 20d cum: %.4f | 60d: %.4f | 120d: %.4f' % (tab.r20.mean(), tab.r60.mean(), tab.r120.mean()))
print('Breadth above MA20: %d/15 | above MA60: %d/15' % (tab.above_ma20.sum(), tab.above_ma60.sum()))
print('Mean 20d ann vol: %.2f%% | median: %.2f%%' % (tab.vol20.mean()*100, tab.vol20.median()*100))
cs_disp = ret.std(axis=1)
print('20d mean daily x-sect dispersion: %.4f%% | 60d: %.4f%% | 5d: %.4f%%' % (cs_disp.tail(20).mean()*100, cs_disp.tail(60).mean()*100, cs_disp.tail(5).mean()*100))
eqw = ret.mean(axis=1)
print('eqw last 5d:', np.round(eqw.tail(5).values, 5))
print('eqw 20d mean daily: %.5f | 60d: %.5f' % (eqw.tail(20).mean(), eqw.tail(60).mean()))

# leader/laggard table (20d / 60d)
print('\n20d leaders/laggards:')
print(tab.r20.sort_values(ascending=False).round(4).to_string())
print('\n60d leaders/laggards:')
print(tab.r60.sort_values(ascending=False).round(4).to_string())

# flat artifacts
flat = (ret.abs() < 1e-12).sum()
print('\nzero-return series last 250d:', flat[flat > 30].to_dict())

# ---------- MACRO ----------
print('\n=== MACRO (through %s) ===' % asof)
mx = {}
for k in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    m = pd.read_csv(f'../persistent/index_data/{k}.csv')
    m['date'] = pd.to_datetime(m['date'])
    m = m[m['date'] <= asof].set_index('date')['close'].sort_index()
    mx[k] = m
    last = m.iloc[-1]
    d20 = m.iloc[-1]/m.iloc[-21]-1 if len(m)>21 else np.nan
    d60 = m.iloc[-1]/m.iloc[-61]-1 if len(m)>61 else np.nan
    print('%s last %.4f | 20d %+.2f%% | 60d %+.2f%%' % (k, last, d20*100, d60*100))
print('VIX last 10:', np.round(mx['VIX'].tail(10).values, 2))
print('VIX 10d ago:', round(mx['VIX'].iloc[-11],2), '20d ago:', round(mx['VIX'].iloc[-21],2) if len(mx['VIX'])>21 else None)

# ---------- FACTOR EXPOSURES ----------
F = {}
F['miner2_20260715_rev_1d']    = -(lpx - lpx.shift(1))
F['miner2_20260715_rev_2d']    = -(lpx - lpx.shift(2))
F['miner2_20260715_rev_3d']    = -(lpx - lpx.shift(3))
F['miner2_20260715_rev_5d']    = -(lpx - lpx.shift(5))
F['miner2_20260715_rev_1d_vs'] = F['miner2_20260715_rev_1d'] / ret.rolling(20).std()
F['miner2_20260715_id_rev_1d'] = -(px/op - 1)
F['miner2_20260715_nbody_1d']  = -(px - op)/(hi - lo)
for n in [1,2,3,5]:
    F[f'miner2_20260715_nclv_{n}d'] = -(px - lo.rolling(n).min()) / (hi.rolling(n).max() - lo.rolling(n).min())
F['mom_120d_skip5'] = px.shift(5)/px.shift(125) - 1
F['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()

vix = mx['VIX']
vix_ret = vix.pct_change()
vixr = vix_ret.reindex(ret.index)
vix20 = (vix/vix.shift(20) - 1).reindex(ret.index)
rb = pd.DataFrame({s: ret[s].rolling(60).cov(vixr) / vixr.rolling(60).var() for s in symbols}, index=ret.index)
F['vix_beta_cond_60x20'] = -rb * vix20
print('\nfactors built t=%.1fs' % (time.time()-t0))

fwd = {}
for h in [1,5,10]:
    fwd[h] = px.shift(-h)/px - 1

def rank_ic_fast(fac, fwd_ret):
    idx = fac.index.intersection(fwd_ret.index)
    f = fac.loc[idx].values; r = fwd_ret.loc[idx].values; dates = idx
    out = np.full(len(dates), np.nan)
    for i in range(len(dates)):
        fi = f[i]; ri = r[i]
        m = ~(np.isnan(fi) | np.isnan(ri))
        if m.sum() < 8: continue
        fx = fi[m]; rx = ri[m]
        fr = np.argsort(np.argsort(fx)); rr = np.argsort(np.argsort(rx))
        fr = fr - fr.mean(); rr = rr - rr.mean()
        denom = np.sqrt((fr*fr).sum() * (rr*rr).sum())
        if denom == 0: continue
        out[i] = float((fr*rr).sum()/denom)
    return pd.Series(out, index=dates).dropna()

results = {}
for fid, fac in F.items():
    entry = {}
    for h in [1,5,10]:
        ic = rank_ic_fast(fac, fwd[h])
        ic60 = ic.iloc[-60:]; ic120 = ic.iloc[-120:]
        entry[f'ic{h}'] = {
            'ic_mean': round(ic60.mean(),4) if len(ic60) else np.nan,
            'icir': round(ic60.mean()/ic60.std()*np.sqrt(252/5),3) if len(ic60)>2 and ic60.std()>0 else 0.0,
            'hit': round((ic60>0).mean(),3) if len(ic60) else np.nan,
            'n': len(ic60),
            'ic_mean_120': round(ic120.mean(),4) if len(ic120) else np.nan,
            'n120': len(ic120)
        }
    results[fid] = entry
    print(fid, 'done t=%.1fs' % (time.time()-t0))

print('\n=== FRESH IC through %s (last 60 obs / 120 obs) ===' % asof)
for fid in F:
    e = results[fid]
    i1, i5, i10 = e['ic1'], e['ic5'], e['ic10']
    print(f'{fid:34s} ic1 {i1["ic_mean"]:+.4f}/{i1["icir"]:+.3f}(h{i1["hit"]}) n{i1["n"]:3d} | ic5 {i5["ic_mean"]:+.4f}/{i5["icir"]:+.3f} | ic10 {i10["ic_mean"]:+.4f}/{i10["icir"]:+.3f}')

print('\n=== QUALITY (q=abs(IC10)*abs(ICIR10), dir=sign(IC10)) ===')
qrows = []
for fid in F:
    e = results[fid]
    i10 = e['ic10']
    ic10 = i10['ic_mean']; icir10 = i10['icir']
    q = abs(ic10)*abs(icir10)
    qrows.append((fid, ic10, icir10, i10['hit'], i10['n'], q, i10['ic_mean_120']))
qrows.sort(key=lambda x: -x[5])
for r in qrows:
    print(f'{r[0]:34s} ic10 {r[1]:+.4f} icir10 {r[2]:+.3f} hit {r[3]} n {r[4]} q {r[5]:.4f} ic120 {r[6]:+.4f}')

json.dump(results, open('_screener_ic.json','w'), indent=1)
print('\nsaved _screener_ic.json t=%.1fs' % (time.time()-t0))
