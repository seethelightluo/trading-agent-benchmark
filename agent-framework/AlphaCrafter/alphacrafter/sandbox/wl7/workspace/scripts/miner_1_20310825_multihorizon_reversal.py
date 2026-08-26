import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
neg=r.clip(upper=0)
d20=np.sqrt(neg.pow(2).rolling(20,min_periods=15).mean())*np.sqrt(20)
d40=np.sqrt(neg.pow(2).rolling(40,min_periods=30).mean())*np.sqrt(40)
s5=r.rolling(5,min_periods=5).sum()/(d20+1e-12)
s20=r.rolling(20,min_periods=20).sum()/(d40+1e-12)
# lag avoids current close information in forward decision
sig=(-(0.6*s5.rank(axis=1,pct=True)+0.4*s20.rank(axis=1,pct=True))).shift(1)
print('rows',len(P),'assets',len(P.columns),'start',P.index.min(),'end',P.index.max())
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'); vals.append(z); ns.append(int(ok.sum())); ds.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(ds)); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
 if h==10: pd.DataFrame({'date':a.index,'ic':a.values,'n':ns}).to_csv('scripts/miner_1_20310825_multihorizon_reversal_ic_10d.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310825_multihorizon_reversal_signal.csv',index=False)
print('coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
