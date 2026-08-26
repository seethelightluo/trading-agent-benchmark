import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5100) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=cl.pct_change(); r20=cl.pct_change(20); r60=cl.pct_change(60); v=r.rolling(40).std()*np.sqrt(252)
cs20=r20.sub(r20.median(axis=1),axis=0); cs60=r60.sub(r60.median(axis=1),axis=0)
sig=(-0.55*cs20/(v+0.05)-0.45*cs60/(v+0.05)).clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(fwd,mask=None):
 vals=[]; ns=[]; dates=sig.index if mask is None else sig.index[mask]
 for dt in dates:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(ok.sum())
 x=pd.Series(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),float((x>0).mean()),float(np.mean(ns)),x
for h in [10,20,40,60]:
 z=calc(cl.shift(-h)/cl-1); print('H',h,'dates',z[0],'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(z[1],z[2],z[3],z[4]))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033-34',sig.index.year.isin([2033,2034]))]:
 z=calc(fwd,mask)
 if z[0]>1: print(name,'dates',z[0],'IC %.6f ICIR %.6f hit %.4f'%(z[1],z[2],z[3]))
 else: print(name,'insufficient dates',z[0])
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).div(len(U)).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20340119_crosssection_residual_reversal_signal.csv',index=False)
