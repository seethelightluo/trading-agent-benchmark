import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); F[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(F).sort_index().ffill(); r=p.pct_change()
# One interpretable idea: drawdown recovery / tail-loss asymmetry, tested at several windows
spec={}
for w in [20,40,60,120]:
 dd=p/p.rolling(w,min_periods=w).min()-1
 vol=r.rolling(20,min_periods=20).std()
 spec[f'recovery_{w}']=dd/vol
 # low downside tail relative to upside tail: negative of downside/upside magnitude
 down=r.where(r<0).rolling(w,min_periods=w).mean().abs()
 up=r.where(r>0).rolling(w,min_periods=w).mean()
 spec[f'tail_asym_{w}']=-(down/(up+1e-12))
for name, sig in spec.items():
 rows=[]
 for t in range(len(r)-1):
  q=pd.concat([sig.iloc[t],r.iloc[t+1]],axis=1).dropna()
  if len(q)>=8: rows.append((r.index[t],q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q),q.iloc[:,0]))
  
  
  
  
 ic=pd.Series([x[1] for x in rows],dtype=float)
 print('\n',name,'dates',len(rows),'avg_n',round(np.mean([x[2] for x in rows]),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 turns=[]; prev=None
 for x in rows:
  rank=x[3].rank(pct=True)
  if prev is not None: turns.append((rank-prev).abs().mean())
  prev=rank
 print(' turnover',round(np.nanmean(turns),5),'coverage',round(np.mean([x[2]/15 for x in rows]),4))
 for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('recent','2026-07-16','2027-01-27')]:
  z=[x[1] for x in rows if str(x[0])[:10]>=a and str(x[0])[:10]<=b]
  if len(z)>1: print(lab,len(z),round(np.mean(z),6),round(np.mean(z)/np.std(z,ddof=1),6),round(np.mean(np.array(z)>0),4))
print('data',len(p),p.index.min(),p.index.max())
