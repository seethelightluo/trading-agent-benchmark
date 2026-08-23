import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={};F={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','open','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').set_index('date');P[s]=x.close;F[s]=-(x.close/x.open-1)
p=pd.DataFrame(P).sort_index().ffill(); r=p.pct_change(); intr=pd.DataFrame(F).reindex(p.index).ffill(); v=r.rolling(20).std(); z=(intr/v).clip(-8,8)
a=[]
for i in range(len(p)-1):
 x=z.iloc[i];y=r.iloc[i+1];ok=x.notna()&y.notna()
 if ok.sum()>=8:a.append(x[ok].corr(y[ok]))
a=pd.Series(a).dropna();print('dates',len(a),'IC %.8f ICIR %.8f hit %.3f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for nm,sel in [('2020-22',pd.Series(a.index)<1095),('recent',pd.Series(range(len(a)))>=len(a)-187)]:
 q=a[sel.values];print(nm,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=z.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20280113_intraday_volnorm_signal.csv',index=False)
