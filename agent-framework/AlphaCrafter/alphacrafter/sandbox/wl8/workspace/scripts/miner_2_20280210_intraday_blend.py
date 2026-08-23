import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={};O={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','open','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').set_index('date');P[s]=x.close;O[s]=-(x.close/x.open-1)
p=pd.DataFrame(P).sort_index().ffill(); r=p.pct_change(); intr=pd.DataFrame(O).reindex(p.index).ffill(); v=r.rolling(20).std()
# Blend two interpretable lagged intraday reversal horizons: vol-normalized current session and 3-session smooth reversal.
sig=(0.65*(intr/v).clip(-8,8)+0.35*intr.rolling(3).mean()).shift(0)
rows=[]
for i in range(len(p)-1):
 z=sig.iloc[i];y=r.iloc[i+1];ok=z.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],z[ok].corr(y[ok]),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for nm,sel in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026',(a.index>='2026-01-01')&(a.index<'2027-01-01')),('2027+',a.index>='2027-01-01'),('recent180',a.index>=a.index.max()-pd.Timedelta(days=260))]:
 z=q[sel];print(nm,len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)) if len(z)>1 else 'NA')
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20280210_intraday_blend_signal.csv',index=False)
