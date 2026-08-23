import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(C).sort_index(); r=np.log(px).diff()
# Candidate: relative 5d reversal, volatility scaled, activated by causal high cross-sectional dispersion.
vol=r.rolling(20,min_periods=15).std()
disp=r.std(axis=1).rolling(20,min_periods=15).mean()
gate=(disp/disp.rolling(120,min_periods=60).median()).clip(.5,2.0)
raw=(-r.rolling(5,min_periods=5).sum()/(vol*np.sqrt(252))).mul(gate,axis=0)
sig=raw.sub(raw.median(axis=1),axis=0)
rows=[]
for h in [1,5,10,20]:
 f=np.log(px.shift(-h)/px); out=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 a=pd.Series([x[1] for x in out],index=[x[0] for x in out]).dropna()
 print('horizon',h,'dates',len(a),'meanN',round(np.mean([x[2] for x in out]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
 if h==5:
  for label,lo,hi in [('2020-24','2020-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31'),('2028-30','2028-01-01','2030-06-12')]:
   q=a.loc[lo:hi]; print('regime',label,'n',len(q),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1),8) if len(q)>1 else np.nan)
print('coverage',round(len(a)/max(1,len(sig)-20),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
sig.to_csv('scripts/miner_1_20300613_dispersion_gated_reversal_signal.csv')
