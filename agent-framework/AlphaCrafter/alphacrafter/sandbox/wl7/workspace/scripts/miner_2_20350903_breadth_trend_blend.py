import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 for f in (get_index_daily_data,get_stock_daily_data):
  try:d=f(s,4200)
  except Exception:d=None
  if d is not None:break
 if d is not None and len(d)>100:d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()});r=cl.pct_change();vol=r.rolling(40,min_periods=20).std(); rows={}
for a in [.5,.6,.7,.8]:
 t20=(cl/cl.shift(20)-1)/(vol*np.sqrt(20));t60=(cl/cl.shift(60)-1)/(vol*np.sqrt(60));m=(a*t20+(1-a)*t60).shift(1); b=(r.rolling(20).sum()>0).mean(axis=1).shift(1); sig=m.mul((2*b-1),axis=0); out=[]
 for dt in sig.index:
  y=cl.shift(-20).loc[dt]/cl.shift(-1).loc[dt]-1;z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q)
 x=pd.Series(out);rows[a]=(len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
 print(a,rows[a], 'recent756',x.tail(756).mean(),x.tail(756).mean()/x.tail(756).std(ddof=1))
# save best candidate structure (a=.7)
a=.7;t20=(cl/cl.shift(20)-1)/(vol*np.sqrt(20));t60=(cl/cl.shift(60)-1)/(vol*np.sqrt(60));b=(r.rolling(20).sum()>0).mean().shift(1);sig=(a*t20+(1-a)*t60).mul(2*b-1,axis=0).shift(1);sig.to_csv('scripts/miner_2_20350903_breadth_trend_blend_signal.csv')
print('assets',len(cl.columns),'coverage',sig.notna().mean().mean())
