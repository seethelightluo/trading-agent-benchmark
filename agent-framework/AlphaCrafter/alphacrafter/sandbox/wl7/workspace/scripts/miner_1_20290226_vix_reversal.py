import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None: P[s]=d.assign(date=pd.to_datetime(d.date).dt.normalize()).drop_duplicates('date').set_index('date').close.loc[:'2029-02-24']
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); vix=get_stock_daily_data('VIX',4000)
# VIX is observation-only but usable as macro signal
v=vix.assign(date=pd.to_datetime(vix.date).dt.normalize()).drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill()
# reversal amplified by unusually high VIX, but shrink to cross-sectional relative signal
z=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
g=(z>0).astype(float)
f=(-r.rolling(5).sum()).mul(g,axis=0); f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20]:
 out=[]; ns=[]
 y=P.shift(-h).div(P)-1
 for i in range(len(P)-h):
  a=f.iloc[i];b=y.iloc[i]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: out.append(a[ok].corr(b[ok],method='spearman'));ns.append(ok.sum())
 a=np.array(out); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),4))
print('active',g.mean())
