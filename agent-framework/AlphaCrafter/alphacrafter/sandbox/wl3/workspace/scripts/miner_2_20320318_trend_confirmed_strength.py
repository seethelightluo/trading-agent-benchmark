import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill()
ret=np.log(px/px.shift(1))
# trend-confirmed excess strength: short relative strength is trusted only when
# its medium trend has same sign; magnitude normalized by trailing vol
med=ret.median(axis=1)
ex20=ret.rolling(20).sum().sub(ret.rolling(20).sum().median(axis=1),axis=0)
# use cross-sectional median of cumulative returns (not future)
ex20=ret.rolling(20).sum().sub(ret.rolling(20).sum().median(axis=1),axis=0)
ex60=ret.rolling(60).sum().sub(ret.rolling(60).sum().median(axis=1),axis=0)
vol=ret.rolling(30).std()*np.sqrt(252)
f=ex20*np.sign(ex60)/vol
# forward 10-day return
fwd=np.log(px.shift(-10)/px)
rows=[]; sig=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
    for s in U:
        sig.append((dt,s,a.get(s,np.nan)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15)
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit', (z.ic>0).mean())
for n in [60,120,252,756]:
 print('recent',n,z.tail(n).ic.mean(),z.tail(n).ic.mean()/z.tail(n).ic.std(ddof=1))
# daily turnover in ranks
sg=pd.DataFrame([(d,s,v) for d,s,v in sig],columns=['date','symbol','v']).pivot(index='date',columns='symbol',values='v')
r=sg.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20320318_trend_confirmed_strength_signal.csv',index=False)
z.to_csv('scripts/miner_2_20320318_trend_confirmed_strength_ic.csv')
