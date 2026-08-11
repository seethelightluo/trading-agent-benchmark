import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# conditional trend: efficiency-weighted 20d momentum, reverse only in high-VIX regime
px={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)==0: d=get_index_daily_data(s,days=4000)
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
v=get_index_daily_data('VIX',days=4000).set_index('date')['close'].astype(float)
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# lag all components one session
r20=close.pct_change(20); vol=ret.rolling(20).std()*np.sqrt(252); eff=r20/(ret.abs().rolling(20).sum()+1e-12)
base=r20/(vol+1e-12)*eff
# high VIX relative to 60d median: reverse trend in stressed regimes
vix_high=(v > v.rolling(60).median()).astype(float).replace(0,1).replace(1,-1) # wrong construction fixed below
reg=pd.Series(np.where(v > v.rolling(60).median(),-1.0,1.0),index=v.index)
sig=base.mul(reg,axis=0).shift(1)
# forward horizons, date intersection
for h in [5,10,20]:
    fwd=close.shift(-h)/close-1
    vals=[]; ns=[]
    for dt in sig.index:
        a=sig.loc[dt]; b=fwd.reindex([dt]).iloc[0] if dt in fwd.index else pd.Series(dtype=float)
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'coverage',np.mean(ns)/15)
# recent regimes
for start in ['2026-01-01','2027-01-01','2028-01-01']:
    fwd=close.shift(-10)/close-1; vals=[]
    for dt in sig.loc[start:].index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    x=pd.Series(vals).dropna(); print(start,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
# artifact
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280323_conditional_trend_signal.csv',index=False)
print('artifact rows',len(out),'assets',len(close.columns),'dates',len(close))
print('regime high pct',reg.eq(-1).mean())
