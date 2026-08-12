import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    try: d=get_index_daily_data(s,days=4100)
    except Exception: d=get_stock_daily_data(s,days=4100)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); P[s]=d.set_index('date')['close'].astype(float).sort_index()
p=pd.DataFrame(P).sort_index(); lr=np.log(p).diff()
# Range/efficiency trend: directional displacement relative to total path, with volatility normalization
ret= p/p.shift(20)-1
path=lr.abs().rolling(20).sum(); eff=ret/(path+1e-12)
vol=lr.rolling(40).std()*np.sqrt(40)
raw=eff/(vol+1e-12)
sig=raw.shift(1).rank(axis=1,pct=True)
rows={h:[] for h in [1,5,10,20]}; dates={h:[] for h in rows}
for dt in sig.index:
    for h in rows:
        f=p.shift(-h)/p-1
        z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            rows[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates[h].append(dt)
for h in rows:
    x=pd.Series(rows[h],index=dates[h]).dropna(); print('%dd dates=%d IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
    if h==1: daily=x
print('avg_names',p.notna().sum(axis=1).mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean(axis=1).dropna().mean())
for yr,g in daily.groupby(daily.index.year): print('year',yr,'IC %.6f n=%d'%(g.mean(),len(g)))
sig.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310123_efficiency_trend_signal.csv',index=False)
