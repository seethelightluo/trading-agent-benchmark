import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   d=f(s,days=5000)
   if d is not None and len(d)>=150:return d
  except: pass
px={}
for s in U:
 d=load(s)
 if d is not None:px[s]=d.set_index('date').close
px=pd.DataFrame(px).sort_index();r=px.pct_change()
# medium-term trend persistence, volatility-normalized and lagged one session
sig=r.rolling(60).sum().shift(1)/r.rolling(40).std().shift(1)
for h in [5,10,20,40]:
 f=px.shift(-h)/px-1; vals=[];ns=[];ds=[]
 for dt in sig.index:
  z=sig.loc[dt];y=f.loc[dt];ok=z.notna()&y.notna()
  if ok.sum()>=8:
   q=z[ok].corr(y[ok],method='spearman')
   if pd.notna(q):vals.append(q);ns.append(ok.sum());ds.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(ds));
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(sig.diff().abs().mean().mean(),6),'recent',round(a[a.index>='2030-01-01'].mean(),6))
print('assets',len(px.columns),'rows',len(px))
