import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2200) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=np.log(px).diff()
# Conditional short-horizon reversal: negate 3-day return only on broad negative breadth,
# otherwise neutral; lag signal one completed day.
breadth=(r<0).mean(axis=1); stress=breadth.shift(1)>=0.60
raw=-r.rolling(3).sum(); cs=raw.sub(raw.mean(axis=1),axis=0)
f=cs.where(stress,0.0).shift(1)
print('instruments',len(px.columns),'dates',len(px),'stress_rate',round(stress.mean(),4))
for h in [1,5,10,20]:
 fr=np.log(px.shift(-h)/px); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>2:
   q=a[ok].corr(b[ok],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(ok.sum());dates.append(dt)
 z=pd.Series(vals,index=dates)
 print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=f.reset_index().rename(columns={'index':'date'});out.to_csv('scripts/miner_2_20340123_stress_reversal_signal.csv',index=False)
