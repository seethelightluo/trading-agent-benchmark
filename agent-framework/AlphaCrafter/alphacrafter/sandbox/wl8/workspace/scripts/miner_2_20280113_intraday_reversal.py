import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cols={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','open','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').set_index('date');cols[s]=-(x.close/x.open-1)
f=pd.DataFrame(cols).sort_index().ffill()
# use same-day intraday reversal as signal, next-day return target
prices={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);prices[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(prices).sort_index().ffill(); r=p.pct_change()
rows=[]
for i in range(len(p)-1):
 z=f.reindex([p.index[i]]).iloc[0];y=r.iloc[i+1];ok=z.notna()&y.notna()
 if ok.sum()>=8:rows.append((p.index[i],z[ok].corr(y[ok]),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'rows',a.n.sum(),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for nm,sel in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026',(a.index>='2026-01-01')&(a.index<'2027-01-01')),('2027',a.index>='2027-01-01'),('recent90',a.index>=a.index.max()-pd.Timedelta(days=140)),('recent180',a.index>=a.index.max()-pd.Timedelta(days=260))]:
 q=a[sel];print(nm,len(q),'IC %.8f ICIR %.8f hit %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20280113_intraday_reversal_signal.csv',index=False)
