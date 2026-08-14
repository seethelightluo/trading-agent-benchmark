import pandas as pd, numpy as np, json

symbols = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
asof = '2034-12-14'

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
print('panel shape', px.shape, 'last', px.index[-1].date())

ret = px.pct_change()

F = {}
F['miner2_20260715_rev_1d']   = -(np.log(px) - np.log(px).shift(1))
F['miner2_20260715_rev_2d']   = -(np.log(px) - np.log(px).shift(2))
F['miner2_20260715_rev_3d']   = -(np.log(px) - np.log(px).shift(3))
F['miner2_20260715_rev_5d']   = -(np.log(px) - np.log(px).shift(5))
F['miner2_20260715_rev_1d_vs']= F['miner2_20260715_rev_1d'] / ret.rolling(20).std()
F['miner2_20260715_id_rev_1d']= -(px/op - 1)
F['miner2_20260715_nbody_1d'] = -(px - op)/(hi - lo)
for n in [1,2,3,5]:
    F[f'miner2_20260715_nclv_{n}d'] = -(px - lo.rolling(n).min()) / (hi.rolling(n).max() - lo.rolling(n).min())
F['mom_120d_skip5'] = px.shift(5)/px.shift(125) - 1
F['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()

vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date'])
vix = vix[vix['date']<=asof].set_index('date')['close'].sort_index()
vix_ret = vix.pct_change()
vix20 = vix/vix.shift(20) - 1

def rolling_beta(y, x, w):
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    yy = y.values; xx = x.values
    for i in range(w, len(y)):
        xw = xx[i-w:i]; yw = yy[i-w:i]
        xc = xw - np.nanmean(xw); yc = yw - np.nanmean(yw, axis=0)
        denom = np.nansum(xc*xc)
        if denom == 0 or np.isnan(denom): continue
        out.iloc[i] = np.nansum(xc[:,None]*yc, axis=0)/denom
    return out

beta60 = rolling_beta(ret, vix_ret.reindex(ret.index), 60)
F['vix_beta_cond_60x20'] = -beta60 * vix20.reindex(ret.index)

fwd = {}
for h in [1,5,10]:
    fwd[h] = px.shift(-h)/px - 1

def rank_ic(fac, fwd_ret):
    out = pd.Series(index=fac.index, dtype=float)
    for dt in fac.index:
        f = fac.loc[dt]; r = fwd_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < 8: continue
        out.loc[dt] = f[mask].rank().corr(r[mask].rank())
    return out.dropna()

results = {}
for fid, fac in F.items():
    entry = {}
    for h in [1,5,10]:
        ic = rank_ic(fac, fwd[h])
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

print('\n=== FRESH IC through 2034-12-14 (last 60 obs / 120 obs) ===')
for fid in F:
    e = results[fid]
    i1, i5, i10 = e['ic1'], e['ic5'], e['ic10']
    print(f'{fid:34s} ic1 {i1["ic_mean"]:+.4f}/{i1["icir"]:+.3f}(h{i1["hit"]}) n{i1["n"]:3d} | ic5 {i5["ic_mean"]:+.4f}/{i5["icir"]:+.3f} | ic10 {i10["ic_mean"]:+.4f}/{i10["icir"]:+.3f}')

json.dump(results, open('_screener_ic_fresh_20341214.json','w'), indent=1)
print('\nsaved _screener_ic_fresh_20341214.json')
