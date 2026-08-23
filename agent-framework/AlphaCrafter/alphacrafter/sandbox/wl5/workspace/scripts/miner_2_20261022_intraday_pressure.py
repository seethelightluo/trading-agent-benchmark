import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
A=[]; total=0
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index(); total+=len(d)
 # Separate intraday return; reversal of recent intraday pressure, volatility normalized.
 oc=d.close/d.open-1
 vol=d.close.pct_change().rolling(20,min_periods=15).std()
 sig=-(oc.rolling(5,min_periods=5).sum()/vol)
 for h in [1,5,10]:
  fr=d.close.shift(-h)/d.close-1
  A.append(pd.DataFrame({'date':d.index,'sig':sig,'fr':fr}).dropna().assign(s=s,h=h))
x=pd.concat(A,ignore_index=True)
for h in [1,5,10]:
 q=x[x.h==h]; ic=[]; ns=[]; dates=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
   ic.append(g.sig.corr(g.fr,method='spearman'));ns.append(len(g));dates.append(dt)
 a=np.array(ic); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[[lo<=d.year<=hi for d in dates]]; print('regime',lo,hi,'n',len(z),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6) if len(z)>1 else np.nan)
q=x[x.h==1]; rank=q.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True); print('coverage',round(len(q)/total,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'assets',q.s.nunique())
q.to_csv('scripts/miner_2_20261022_intraday_pressure_signal.csv',index=False)
