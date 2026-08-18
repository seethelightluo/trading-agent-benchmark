import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-31'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x.loc[:END]
def fac(x):
 r=x.close.pct_change(); return r.rolling(5).sum()-r.rolling(20).sum()/4
for h in [5,10,20]:
 rec=[]
 for s,x in D.items():
  f=fac(x); y=x.close.shift(-h)/x.close-1
  for d in x.index:
   if pd.notna(f.loc[d]) and pd.notna(y.loc[d]):rec.append((d,s,f.loc[d],y.loc[d]))
 a=pd.DataFrame(rec,columns=['d','s','f','y']); z=[];ns=[]
 for d,g in a.groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.array(z);print('accel h',h,'dates',len(z),'N',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0),'cov',a.s.nunique()/15)
 if h==10:a.to_csv('scripts/miner_2_20271231_momentum_acceleration_signal.csv',index=False)
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31')]:
 g0=a[(a.d>=lo)&(a.d<=hi)]; zz=[]
 for d,g in g0.groupby('d'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:zz.append(spearmanr(g.f,g.y).statistic)
 zz=np.array(zz);print(label,len(zz),np.mean(zz),np.mean(zz)/np.std(zz,ddof=1))
