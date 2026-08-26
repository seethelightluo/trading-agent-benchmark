import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); rows=[]
# Dispersion-scaled relative reversal: fade each asset's 5d residual return,
# scaled by its own 20d realized volatility; higher score means cheaper residual per unit risk.
for i,t in enumerate(p.index):
 if i<25 or i+5>=len(p): continue
 ret5=p.iloc[i]/p.iloc[i-5]-1
 vol20=r.iloc[i-20:i].std()*np.sqrt(20)
 residual=ret5-ret5.median()
 sig=-residual/(vol20+1e-12)
 f=p.iloc[i+5]/p.iloc[i]-1
 q=pd.concat([sig,f],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((t,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
A=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=A.ic.to_numpy()
def stat(z): return (len(z),float(z.mean()),float(z.mean()/(z.std(ddof=1)+1e-12)*np.sqrt(252)),float((z>0).mean()))
print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(p),'valid_dates',len(A),'mean_n',A.n.mean())
print('5d',stat(x));
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-03-07'),('2028','2029-03-07'),('2028-06-01','2029-03-07')]:
 z=A.loc[a:b].ic.to_numpy(); print(a,b,stat(z) if len(z) else None)
for h in [1,10,20]:
 rows2=[]
 for i,t in enumerate(p.index):
  if i<25 or i+h>=len(p): continue
  ret5=p.iloc[i]/p.iloc[i-5]-1; vol20=r.iloc[i-20:i].std()*np.sqrt(20); sig=-(ret5-ret5.median())/(vol20+1e-12); f=p.iloc[i+h]/p.iloc[i]-1
  q=pd.concat([sig,f],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows2.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print(str(h)+'d',stat(np.array(rows2)))
