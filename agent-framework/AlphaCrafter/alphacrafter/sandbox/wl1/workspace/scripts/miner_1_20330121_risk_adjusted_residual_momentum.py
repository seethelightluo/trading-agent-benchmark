import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().replace([np.inf,-np.inf],np.nan); R=P.pct_change()
raw=P.pct_change(60); vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
f=raw/vol; f=f.sub(f.mean(axis=1),axis=0).shift(1).replace([np.inf,-np.inf],np.nan)
fr=P.shift(-10)/P-1; ics=[]; rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  c=x[ok].corr(y[ok])
  if np.isfinite(c): ics.append(c); rows.append((dt,c,int(ok.sum())))
a=np.asarray(ics); r=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(r)):
 ok=r.iloc[i].notna()&r.iloc[i-1].notna()
 if ok.sum()>=8: turn.append((r.iloc[i][ok]-r.iloc[i-1][ok]).abs().mean())
print('candidate risk_adjusted_residual_momentum_60d')
print('dates',len(a),'avg_instruments',np.mean([z[2] for z in rows]),'coverage',P.notna().mean().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(turn))
for name,lo,hi in [('2020-23','2020','2023-12-31'),('2024-26','2024','2026-12-31'),('2027-29','2027','2029-12-31'),('2030-33','2030','2033-01-07')]:
 q=np.array([c for d,c,n in rows if str(d)>=lo and str(d)<=hi]); print(name,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame([(d.strftime('%Y-%m-%d'),s,float(f.loc[d,s]) if pd.notna(f.loc[d,s]) else np.nan) for d in f.index for s in U],columns=['date','symbol','signal']); out.to_csv('scripts/miner_1_20330121_risk_adjusted_residual_momentum_signal.csv',index=False)
