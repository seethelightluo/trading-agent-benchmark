import os, json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
px={}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv'); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index()
# signal known at t: trailing 5d return, volatility-scaled, then evaluate t+H
ret=p.pct_change(); sig=(p/p.shift(5)-1)/(ret.rolling(20).std()*np.sqrt(5))
for h in [1,5,10,20]:
  f=sig.shift(1); fw=p.shift(-h)/p-1
  ics=[]; n=[]; turnovers=[]
  for dt in f.index:
   x=f.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
   if ok.sum()>=8:
    ics.append(spearmanr(x[ok],y[ok]).statistic); n.append(ok.sum())
  ic=np.array(ics); print('H',h,'dates',len(ic),'meanN',round(np.mean(n),2),'IC',round(ic.mean(),5),'ICIR',round(ic.mean()/ic.std(ddof=1),5),'hit',round((ic>0).mean(),4))
# 10-day rank turnover proxy
r=sig.rank(axis=1,pct=True); to=(r-r.shift(10)).abs().mean(axis=1).dropna(); print('turnover10',round(to.mean(),4),'coverage',round(sig.notna().mean().mean(),4),'rows',len(p),'assets',len(assets),'end',p.index.max().date())
# regimes, H=5
f=sig.shift(1); fw=p.shift(-5)/p-1
for label,lo,hi in [('2020-2024','2020','2025'),('2025-2027','2025','2028'),('2028-2029','2028','2030')]:
 z=[]
 for dt in f.index:
  if str(dt.date())<lo+'-01-01' or str(dt.date())>=hi+'-01-01': continue
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],fw.loc[dt,ok]).statistic)
 print(label,len(z),round(np.mean(z),5) if z else None,round(np.mean(z)/np.std(z,ddof=1),5) if len(z)>1 else None)
