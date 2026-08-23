import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); rows=[]
for t in p.index:
 rr=p.loc[t]/p.shift(20).loc[t]-1; down=r.loc[:t].tail(40).clip(upper=0).std()*np.sqrt(40); hi=p.loc[:t].tail(60).max(); rec=(p.loc[t]/hi).clip(0,1)
 f=(rr/down*rec).replace([np.inf,-np.inf],np.nan)
 for s in U:
  if s in f and pd.notna(f[s]): rows.append((t,s,f[s]))
F=pd.DataFrame(rows,columns=['date','symbol','factor']); F.to_csv('scripts/miner_2_20290125_recovery_trend_signal.csv',index=False)
for h in [5,10,20]:
 fw=p.shift(-h)/p-1; a=[]; cov=[]
 for t,g in F.groupby('date'):
  x=g.set_index('symbol').factor; y=fw.loc[t].reindex(x.index); z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.factor.corr(z.y,method='spearman'));cov.append(len(z)/15)
 a=pd.Series(a).dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'obs',len(a),'meanN',np.mean(cov))
q=F.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('rows',len(F),'dates',F.date.nunique(),'turnover',q.diff().abs().mean().mean(),'coverage',F.groupby('date').symbol.nunique().mean()/15)
for name,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31'),('recent','2028-01-01','2029-01-25')]:
 fw=p.shift(-10)/p-1;a=[]
 for t,g in F[(F.date>=lo)&(F.date<=hi)].groupby('date'):
  x=g.set_index('symbol').factor;y=fw.loc[t].reindex(x.index);z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.factor.corr(z.y,method='spearman'))
 a=pd.Series(a).dropna();print(name,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
