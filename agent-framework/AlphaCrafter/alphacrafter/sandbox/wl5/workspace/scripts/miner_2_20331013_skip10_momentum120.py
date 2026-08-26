import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Medium-term trend excluding short reversal window; volatility normalized.
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index()
ret=px.pct_change()
# 120-day trend, skip most recent 10 days, scaled by 40d realized vol
signal=(px.shift(10)/px.shift(130)-1)/(ret.rolling(40).std()*np.sqrt(40))
# avoid extreme scale, rank IC insensitive
fwd=px.shift(-10)/px-1
rows=[]; artifact=[]
for dt in signal.index:
    a=signal.loc[dt]; b=fwd.loc[dt]
    ok=a.notna()&b.notna(); n=int(ok.sum())
    if n>=8:
        ic=spearmanr(a[ok],b[ok]).statistic
        rows.append((dt,ic,n))
        for s in U:
            if ok.get(s,False): artifact.append((dt,s,float(a[s]),float(b[s])))
r=pd.DataFrame(rows,columns=['date','ic','n'])
# use online-era and broad post-warmup, 2026 onward
r=r[r.date>=pd.Timestamp('2026-07-28')]
mean=r.ic.mean(); sd=r.ic.std(ddof=1)
print('dates',len(r),'period',r.date.min().date(),r.date.max().date(),'mean_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',mean,'ICIR',mean/sd*np.sqrt(252) if sd else np.nan,'hit',(r.ic>0).mean(),'sd',sd)
# 3 regimes
for name,lo,hi in [('2026-27','2026-07-28','2027-12-31'),('2028-29','2028-01-01','2029-12-31'),('2030-33','2030-01-01','2033-10-12')]:
 q=r[(r.date>=lo)&(r.date<=hi)]
 print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan,(q.ic>0).mean())
# turnover: rank changes by date across overlapping available dates
z=signal.loc[r.date].rank(axis=1,pct=True); turnover=z.diff().abs().mean(axis=1).mean()
print('turnover',turnover)
pd.DataFrame(artifact,columns=['date','symbol','signal','forward_return']).to_csv('scripts/miner_2_20331013_skip10_momentum120_signal.csv',index=False)
