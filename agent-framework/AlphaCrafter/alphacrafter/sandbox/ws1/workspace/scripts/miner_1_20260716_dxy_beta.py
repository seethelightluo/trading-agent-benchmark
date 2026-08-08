import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,3000)
   if d is not None:return d
  except (FileNotFoundError,KeyError): pass
 return None
macro=fetch('DXY'); macro['date']=pd.to_datetime(macro['date']); m=macro.set_index('date')['close'].pct_change()
prices={}
for s in U:
 d=fetch(s)
 if d is not None: d['date']=pd.to_datetime(d['date']); prices[s]=d.set_index('date')['close']
P=pd.DataFrame(prices); R=P.pct_change(); m=m.reindex(R.index)
for window in [40,60,120]:
 fac=pd.DataFrame(index=R.index); cov=R.rolling(window).cov(m); var=m.rolling(window).var()
 for s in prices: fac[s]=-cov[s]/var
 fwd=R.shift(-1); ics=[]; counts=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
 ranks=fac.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).mean(); a=np.array(ics)
 recent=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if str(dt)[:4]>='2025' and len(z)>=8: recent.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(window,'assets',len(prices),'N',len(a),'meanN',np.mean(counts),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turn',turn,'coverage',fac.notna().sum().sum()/(len(fac)*max(1,len(prices))),'recent',np.nanmean(recent))
