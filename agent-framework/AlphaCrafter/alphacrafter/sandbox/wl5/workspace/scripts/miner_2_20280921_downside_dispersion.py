import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,3000)
    if x is not None and len(x)>100: D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
neg=r.clip(upper=0).rolling(5).mean(); disp=neg.std(axis=1); th=disp.rolling(60).quantile(.7); shock=disp>th; sig=-r.rolling(5).sum(); fwd=p.shift(-10)/p-1
rows=[]
for dt in p.index:
    if not shock.get(dt,False): continue
    z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n'])
print('data',len(p),len(D),'dates',len(x),'range',x.date.min(),x.date.max())
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'avgN',x.n.mean())
for cutoff in ['2024-01-01','2026-07-16','2027-01-01','2028-01-01']:
 y=x[x.date>=cutoff]; print(cutoff,len(y),y.ic.mean(),y.ic.mean()/y.ic.std(ddof=1) if len(y)>1 else np.nan)
for h in [5,10,15,20]:
 z=[]
 for dt in p.index:
  if not shock.get(dt,False):continue
  q=pd.concat([sig.loc[dt],(p.shift(-h)/p-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('decay',h,len(z),np.nanmean(z))
out=sig.where(shock).stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280921_downside_dispersion_signal.csv',index=False)
