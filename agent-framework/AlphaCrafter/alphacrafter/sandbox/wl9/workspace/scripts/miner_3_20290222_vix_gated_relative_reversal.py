import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.drop_duplicates('date').set_index('date')['close'].reindex(p.index).ffill()
r=p.pct_change(); rows=[]
# VIX-gated relative reversal: fade 5d asset return relative to cross-sectional median only
# when VIX is above its trailing 60-session median; otherwise use neutral zero signal.
for i,t in enumerate(p.index):
 if i<70 or i+5>=len(p): continue
 vv=v.iloc[i]
 if not np.isfinite(vv) or vv<=v.iloc[i-60:i].median(): continue
 rr=p.iloc[i]/p.iloc[i-5]-1; sig=-(rr-rr.median()); f=p.iloc[i+5]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']); x=A.ic.to_numpy()
print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'all_dates',len(p),'gated_dates',len(A),'mean_n',A.n.mean())
print('ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
for label, lo, hi in [('2020-2023','2020','2024'),('2024-2026','2024','2027'),('2027-2029','2027','2030')]:
 q=A[(A.date>=lo)&(A.date<hi)].ic
 print(label,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
