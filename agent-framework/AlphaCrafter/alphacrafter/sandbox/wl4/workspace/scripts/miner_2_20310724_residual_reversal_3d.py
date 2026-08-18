import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_index_daily_data(s,3200)
 except Exception:
  try:x=get_stock_daily_data(s,3200)
  except:continue
 if x is not None and len(x)>100:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index();r=P.pct_change();res=r.rolling(3,min_periods=2).sum().sub(r.rolling(3,min_periods=2).sum().median(axis=1),axis=0);f=(-res).shift(1)
ics={h:[] for h in [3,5,10]}; ns=[]
for i in range(len(P)-10):
 z=f.iloc[i];n=z.notna()&P.iloc[i].notna();
 if n.sum()>=8:
  ns.append(n.sum())
  for h in ics:
   fr=P.iloc[i+h]/P.iloc[i]-1;q=n&fr.notna()
   if q.sum()>=8:ics[h].append(z[q].corr(fr[q],method='spearman'))
for h,a in ics.items():
 a=np.array(a);print('H',h,'obs',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('dates',len(P),'assets',len(D),'avgN',np.mean(ns),'coverage',np.mean(ns)/15)
for w in [365,730,1095]:
 a=np.array(ics[5][-w:]);print('recent',w,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))
