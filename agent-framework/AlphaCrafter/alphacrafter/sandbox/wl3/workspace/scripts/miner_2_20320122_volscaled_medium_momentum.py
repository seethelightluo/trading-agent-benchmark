import os, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volatility-scaled medium-term momentum, with signal direction damped in stressed cross-asset volatility
# all inputs lagged at date t; forward return t+1..t+10
series={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)==0: continue
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
    series[s]=np.log(d['close'].astype(float)).diff()
R=pd.DataFrame(series).sort_index()
# 20d return divided by 20d realized vol; cross-sectional factor. In high aggregate vol, attenuate trend
mom=R.rolling(20).sum()
vol=R.rolling(20).std()*np.sqrt(20)
base=mom/vol.replace(0,np.nan)
agg=R.mean(axis=1).rolling(20).std()
# stress is lagged aggregate vol relative to 120d median; trend signal is attenuated, not flipped
stress=(agg/agg.rolling(120).median()).clip(0.5,2.0)
f=base.div(stress,axis=0)
rows=[]; daily=[]
for i,t in enumerate(R.index):
    if i+10>=len(R): break
    x=f.loc[t]; y=R.iloc[i+1:i+11].sum()
    z=pd.concat([x,y],axis=1).dropna(); n=len(z)
    if n>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((t,ic,n))
        daily.append({'date':t.strftime('%Y-%m-%d'),'factor':float(x.get('SPX',np.nan)),'forward_return':float(y.get('SPX',np.nan))})
D=pd.DataFrame(rows,columns=['date','ic','n'])
mu=D.ic.mean(); sd=D.ic.std(ddof=1); icir=mu/sd*np.sqrt(252) if sd>0 else np.nan
# rank turnover based on consecutive factor ranks
turn=[]; prev=None
for t in D.date:
    r=f.loc[t].rank(pct=True)
    if prev is not None: turn.append((r-prev).abs().mean())
    prev=r
recent=D[D.date>=pd.Timestamp('2030-01-01')]
print('dates',len(D),'avg_n',D.n.mean(),'coverage',D.n.mean()/15,'IC',mu,'ICIR',icir,'hit', (D.ic>0).mean(),'turnover',np.mean(turn))
print('recent dates',len(recent),'IC',recent.ic.mean(),'ICIR',recent.ic.mean()/recent.ic.std(ddof=1)*np.sqrt(252))
for a,b in [('2020-01-01','2025-12-31'),('2026-01-01','2029-12-31'),('2030-01-01','2032-01-22')]:
 q=D[(D.date>=a)&(D.date<=b)]; print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>2 else np.nan)
out='scripts/miner_2_20320122_volscaled_medium_momentum_signal.csv'
pd.DataFrame(daily).to_csv(out,index=False)
print('artifact',out)
