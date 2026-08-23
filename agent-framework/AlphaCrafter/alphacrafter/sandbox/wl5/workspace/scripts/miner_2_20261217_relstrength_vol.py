import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.drop_duplicates('date').set_index('date').sort_index()
# Candidate: cross-asset relative volatility carry. 20d realized volatility rank, gated by
# whether the asset's 5d return agrees with the cross-sectional median. Low-vol assets
# with positive recent relative strength are favored; signal is formed after close.
A=[]; total=0
for s,d in D.items():
 total+=len(d); r=d.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 med=r.groupby(d.index).transform('median') if False else None
 # date-aligned cross-sectional median is constructed after concatenation
 A.append(pd.DataFrame({'date':d.index,'r5':r.rolling(5,min_periods=5).sum(),'vol':vol,'s':s,
                        'fr':d.close.shift(-1)/d.close-1}).dropna())
x=pd.concat(A,ignore_index=True)
x['med5']=x.groupby('date')['r5'].transform('median')
# signed relative strength multiplied by inverse volatility; interpretable and bounded
x['sig']=((x.r5-x.med5)/(x.vol+1e-12)).clip(-8,8)
ics=[];ns=[];dates=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
  z=g.sig.corr(g.fr,method='spearman')
  if np.isfinite(z):ics.append(z);ns.append(len(g));dates.append(dt)
a=np.array(ics)
print('dates',len(a),'avg_n',round(np.mean(ns),2),'assets',x.s.nunique(),'coverage',round(len(x)/total,4))
print('IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=a[[lo<=d.year<=hi for d in dates]]
 print('regime',lo,hi,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
for h in [5,10]:
 q=[]
 for s,d in D.items():
  r=d.close.pct_change();v=r.rolling(20,min_periods=15).std();r5=r.rolling(5,min_periods=5).sum()
  q.append(pd.DataFrame({'date':d.index,'sig':((r5-r5.groupby(d.index).transform('median'))/(v+1e-12)).clip(-8,8),'fr':d.close.shift(-h)/d.close-1}).dropna())
 q=pd.concat(q); z=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.sig.nunique()>1:z.append(g.sig.corr(g.fr,method='spearman'))
 z=np.array(z);print('horizon',h,'dates',len(z),'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6))
r=x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True)
print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
x[['date','s','sig','fr']].to_csv('scripts/miner_2_20261217_relstrength_vol_signal.csv',index=False)
