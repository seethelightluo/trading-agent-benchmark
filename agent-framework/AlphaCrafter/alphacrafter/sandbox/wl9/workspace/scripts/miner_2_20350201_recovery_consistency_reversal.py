import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Factor favors assets with medium-term losses but persistent, orderly recovery attempts:
# negative 60d return, multiplied by fraction of positive days in last 20d, volatility scaled.
frames={}
for s in U:
    d=get_index_daily_data(s, days=2600)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date')
        frames[s]=d.close.astype(float)
p=pd.DataFrame(frames).sort_index().dropna(how='all')
ret=p.pct_change()
# lag all signal ingredients by one day; forward 60 return
r60=p.pct_change(60); sig=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
pos20=(ret>0).rolling(20,min_periods=15).mean()
vol60=ret.rolling(60,min_periods=40).std()*np.sqrt(252)
sig= -r60 * (0.5+pos20) / (vol60+1e-8)
# cross-sectional daily IC, forward return from t+1 close to t+60 close
fwd=p.shift(-60)/p.shift(-1)-1
rows=[]
for i in range(len(p)-60):
    dt=p.index[i]
    x=sig.iloc[i-1] if i>0 else pd.Series(dtype=float)
    y=fwd.iloc[i]
    z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.x.corr(z.y)))
r=pd.DataFrame(rows,columns=['date','n','ic'])
# report regimes and turnover/coverage
ics=r.ic
print('DATES',len(r),'AVG_N',r.n.mean(),'MIN_N',r.n.min(),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'HIT', (ics>0).mean())
print('REGIMES')
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2035-12-31')]:
 q=r[(r.date>=a)&(r.date<=b)]
 if len(q): print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan,(q.ic>0).mean())
# rank turnover among valid cross sections
ranks=sig.rank(axis=1,pct=True); common=ranks.notna() & ranks.shift(1).notna()
turn=(ranks-ranks.shift(1)).abs().where(common).mean(axis=1).mean()
print('COVERAGE',sig.notna().mean().mean(),'TURNOVER',turn)
# save artifact for audit
out=sig.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20350201_recovery_consistency_reversal_signal.csv')
