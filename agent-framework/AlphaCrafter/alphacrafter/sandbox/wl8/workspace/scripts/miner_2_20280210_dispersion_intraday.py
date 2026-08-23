import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={};O={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  x=d[['date','open','close']].copy();x.date=pd.to_datetime(x.date);x=x.drop_duplicates('date').set_index('date');P[s]=x.close;O[s]=-(x.close/x.open-1)
p=pd.DataFrame(P).sort_index().ffill();r=p.pct_change();intr=pd.DataFrame(O).reindex(p.index).ffill();v=r.rolling(20).std()
# Regime-conditioned intraday reversal: emphasize reversal when lagged cross-asset dispersion is high.
disp=r.std(axis=1).rolling(20).mean(); z=(intr/v).clip(-8,8).mul((disp/disp.rolling(60).median()).shift(1),axis=0).clip(-12,12)
rows=[]
for i in range(len(p)-1):
 x=z.iloc[i];y=r.iloc[i+1];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((p.index[i],x[ok].corr(y[ok]),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),2),'coverage',round(z.notna().sum().sum()/z.size,4),'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for nm,sel in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026',(a.index>='2026-01-01')&(a.index<'2027-01-01)),('2027+',a.index>='2027-01-01'),('recent180',a.index>=a.index.max()-pd.Timedelta(days=260))]:
 x=q[sel];print(nm,len(x),'IC %.8f ICIR %.8f'%(x.mean(),x.mean()/x.std(ddof=1)) if len(x)>1 else 'NA')
