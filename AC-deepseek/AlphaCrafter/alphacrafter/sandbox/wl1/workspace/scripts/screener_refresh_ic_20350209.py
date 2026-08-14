import pandas as pd, numpy as np, json, time
t0 = time.time()

symbols = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
asof = '2035-02-08'

dfs = {}
for s in symbols:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= asof].set_index('date').sort_index()
    dfs[s] = df[['open','high','low','close']]

px = pd.DataFrame({s: dfs[s]['close'] for s in symbols})
op = pd.DataFrame({s: dfs[s]['open'] for s in symbols})
hi = pd.DataFrame({s: dfs[s]['high'] for s in symbols})
lo = pd.DataFrame({s: dfs[s]['low'] for s in symbols})
print('panel shape', px.shape, 'last', px.index[-1].date(), 't=%.1fs' % (time.time()-t0))

ret = px.pct_change()
lpx = np.log(px)

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

vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date'])
vix = vix[vix['date']<=asof].set_index('date')['close'].sort_index()
vix_ret = vix.pct_change()
vix20 = vix/vix.shift(20) - 1
rb = ret.rolling(60).cov(vix_ret.reindex(ret.index)) / vix_ret.reindex(ret.index).rolling(60).var()
F['vix_beta_cond_60x20'] = -rb * vix20.reindex(ret.index)
print('factors built t=%.1fs' % (time.time()-t0))

fwd = {}
for h in [1,5,10]:
    fwd[h] = px.shift(-h)/px - 1

def rank_ic_fast(fac, fwd_ret):
    idx = fac.index.intersection(fwd_ret.index)
    f = fac.loc[idx].values
    r = fwd_ret.loc[idx].values
    dates = idx
    out = np.full(len(dates), np.nan)
    for i in range(len(dates)):
        fi = f[i]; ri = r[i]
        m = ~(np.isnan(fi) | np.isnan(ri))
        if m.sum() < 8:
            continue
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
        ic60 = ic.iloc[-60:]
        ic120 = ic.iloc[-120:]
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

print('\n=== FRESH IC through 2035-02-08 (last 60 obs / 120 obs) ===')
for fid in F:
    e = results[fid]
    i1, i5, i10 = e['ic1'], e['ic5'], e['ic10']
    print(f'{fid:34s} ic1 {i1["ic_mean"]:+.4f}/{i1["icir"]:+.3f}(h{i1["hit"]}) n{i1["n"]:3d} | ic5 {i5["ic_mean"]:+.4f}/{i5["icir"]:+.3f} | ic10 {i10["ic_mean"]:+.4f}/{i10["icir"]:+.3f}')

json.dump(results, open('_screener_ic.json','w'), indent=1)
print('\nsaved _screener_ic.json t=%.1fs' % (time.time()-t0))
