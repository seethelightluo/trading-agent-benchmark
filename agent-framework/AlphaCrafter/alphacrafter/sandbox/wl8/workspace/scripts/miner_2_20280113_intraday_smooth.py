import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; O={};C={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','open','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').set_index('date');O[s]=-(x.close/x.open-1);C[s]=x.close
p=pd.DataFrame(C).sort_index().ffill(); f=pd.DataFrame(O).sort_index().ffill().rolling(3).mean();r=p.pct_change();a=[]
for i in range(len(p)-1):
 z=f.iloc[i];y=r.iloc[i+1];ok=z.notna()&y.notna()
 if ok.sum()>=8:a.append(z[ok].corr(y[ok]))
a=pd.Series(a).dropna();print('dates',len(a),'IC %.8f ICIR %.8f hit %.3f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [1,3,5]:
 b=[]
 for i in range(len(p)-h):
  z=f.iloc[i];y=p.iloc[i+h]/p.iloc[i]-1;ok=z.notna()&y.notna()
  if ok.sum()>=8:b.append(z[ok].corr(y[ok]))
 b=pd.Series(b).dropna();print('h',h,'IC %.8f ICIR %.8f'%(b.mean(),b.mean()/b.std(ddof=1)))
