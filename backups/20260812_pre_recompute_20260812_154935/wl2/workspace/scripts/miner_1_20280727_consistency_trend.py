import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(s,days=4000)
   if x is not None and len(x)>100:return x
  except:pass
 return None
def main():
 d={s:fetch(s) for s in S}; d={s:x for s,x in d.items() if x is not None}; p=pd.concat({s:x.set_index('date').close for s,x in d.items()},axis=1).sort_index(); r=p.pct_change()
 # consistency: fraction of positive sessions, scaled by net trend, known at t
 consistency=(r.gt(0).rolling(40,min_periods=30).mean()-0.5)*2
 trend=p.pct_change(40)
 sig=consistency*trend
 for h in [1,5,10,20]:
  f=p.shift(-h)/p-1; out=[]
  for dt in sig.index:
   a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(a)>=8:out.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
  q=pd.Series(out).dropna();print('h',h,'obs',len(q),'avgN',len(d),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
 z=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna();print('turn',z.mean())
if __name__=='__main__':main()
