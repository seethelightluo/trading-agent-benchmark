import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: d=get_index_daily_data(s,3000)
 except: d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>300: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); m=r.mean(axis=1)
rows=[]
for i in range(80,len(p)-10):
 f=pd.Series(index=p.columns,dtype=float); rr=r.iloc[max(0,i-40):i]
 for s in p.columns:
  b=rr[s].cov(m.iloc[max(0,i-40):i])/m.iloc[max(0,i-40):i].var()
  res=rr[s].iloc[-3:].sum()-b*m.iloc[max(0,i-3):i].sum(); vol=rr[s].std()
  f[s]=-res/(vol*np.sqrt(3)) if vol>1e-8 else np.nan
 fr=p.iloc[i+10]/p.iloc[i]-1; z=pd.concat([f.rename('f'),fr.rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],z.f.corr(z.y),len(z)))
ics=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); ns=pd.Series([x[2] for x in rows],index=ics.index)
print('dates',len(ics),'avgN',ns.mean(),'coverage',ns.mean()/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean())
for n in [120,252,504]:
 q=ics.tail(n); print('recent',n,q.mean(),q.mean()/q.std(ddof=1),len(q))
for s in ['2020','2022','2024','2026','2028','2030','2032','2034']:
 q=ics[ics.index.astype(str).str.startswith(s)]
 if len(q): print(s,len(q),q.mean(),q.mean()/q.std(ddof=1))
ics.to_csv('scripts/miner_2_20341110_residual_reversal3_ic.csv')
