import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]; total=0
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index(); total+=len(d)
 r=d.close.pct_change(); w=30
 # downside asymmetry: fraction of realized variance attributable to negative returns, inverted
 dn=r.clip(upper=0).pow(2).rolling(w,min_periods=20).sum()
 allv=r.pow(2).rolling(w,min_periods=20).sum()
 sig=-(dn/allv)
 fr=d.close.shift(-1)/d.close-1
 A.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s))
x=pd.concat(A,ignore_index=True)
for h in [1,5,10]:
 q=[]
 for s in U:
  d=get_stock_daily_data(s,4000)
  if d is None: continue
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();r=d.close.pct_change();dn=r.clip(upper=0).pow(2).rolling(30,min_periods=20).sum();av=r.pow(2).rolling(30,min_periods=20).sum();sig=-(dn/av);fr=d.close.shift(-h)/d.close-1
  q.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s))
 z=pd.concat(q); ic=[];ns=[];ds=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
   v=g.sig.corr(g.fr,method='spearman')
   if np.isfinite(v):ic.append(v);ns.append(len(g));ds.append(dt)
 a=np.array(ic);print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  b=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6))
r=x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True)
print('coverage',round(len(x)/total,4),'turnover',round(r.diff().abs().mean(axis=1).mean(),4),'assets',x.s.nunique())
x.to_csv('scripts/miner_2_20261217_downside_asymmetry_signal.csv',index=False)
