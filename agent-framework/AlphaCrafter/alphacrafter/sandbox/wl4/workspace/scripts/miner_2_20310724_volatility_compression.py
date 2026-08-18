import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:x=get_index_daily_data(s,days=3200)
 except Exception:
  try:x=get_stock_daily_data(s,days=3200)
  except Exception: continue
 if x is not None and len(x)>100:
  x=x.copy();x['date']=pd.to_datetime(x['date']);D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index();r=P.pct_change();v20=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=40).std();trend=r.rolling(20,min_periods=15).sum();f=(-v20/v60+.15*trend/(v20*np.sqrt(20))).shift(1)
ics={h:[] for h in [5,10,20]};ns=[];turns=[]
for i in range(len(P)-20):
 z=f.iloc[i];n=z.notna()&P.iloc[i].notna();ns.append(int(n.sum()))
 if n.sum()>=8:
  for h in ics:
   fr=P.iloc[i+h]/P.iloc[i]-1;q=n&fr.notna()
   if q.sum()>=8:ics[h].append(z[q].corr(fr[q],method='spearman'))
for i in range(10,len(f)):
 a=f.iloc[i-10];b=f.iloc[i];q=a.notna()&b.notna()
 if q.sum()>=8:turns.append(1-a[q].rank().corr(b[q].rank()))
print('dates',len(P),'assets',len(D),'observations',len(ics[10]),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15))
for h,a in ics.items():
 a=np.array(a);print('H',h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('turnover',np.nanmean(turns))
for w in [365,730,1095]:
 a=np.array(ics[10][-w:]);print('recent',w,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1))
