import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]; total=0
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();total+=len(d)
 o=d.open/d.close.shift(1)-1; i=d.close/d.open-1; v=d.close.pct_change().rolling(20,min_periods=15).std()
 # One-session overnight/intraday divergence, volatility scaled, with no multi-day smoothing.
 sig=(o-i)/v
 fr=d.close.shift(-1)/d.close-1
 A.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s))
x=pd.concat(A,ignore_index=True);ics=[];ns=[];ds=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
  z=g.sig.corr(g.fr,method='spearman')
  if np.isfinite(z):ics.append(z);ns.append(len(g));ds.append(dt)
a=np.array(ics);print('dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',len(x)/total)
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=a[[lo<=d.year<=hi for d in ds]];print('regime',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
r=x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean(),'assets',x.s.nunique())
x.to_csv('scripts/miner_2_20261022_overnight_intraday1_signal.csv',index=False)
