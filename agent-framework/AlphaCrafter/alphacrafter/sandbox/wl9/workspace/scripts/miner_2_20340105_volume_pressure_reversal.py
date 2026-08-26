import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5100) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
vol=pd.DataFrame({s:d.set_index('date')['volume'] for s,d in D.items() if d is not None}).reindex(cl.index).ffill()
r=cl.pct_change(); r20=cl.pct_change(20); r60=cl.pct_change(60)
v20=r.rolling(20).std()*np.sqrt(252); v60=r.rolling(60).std()*np.sqrt(252)
# Reversal is amplified when recent selling occurs on unusually heavy volume, a simple capitulation proxy.
vs=vol/(vol.rolling(60).median()+1e-12)
pressure=(vs.rolling(20).mean()-1).clip(-1,2)
sig=(-.45*r20/(v20+.05)-.55*r60/(v60+.05))*(1+0.30*pressure.clip(-1,1))
sig=sig.clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(fwd, idx=None):
 xs=[]; ns=[]
 for dt in sig.index if idx is None else sig.index[idx]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q); ns.append(ok.sum())
 x=pd.Series(xs); return len(x),x.mean(),x.mean()/x.std(ddof=1), (x>0).mean(),np.mean(ns),x
for h in [10,20,40,60]:
 z=calc(cl.shift(-h)/cl-1); print('H',h,'dates',z[0],'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(z[1],z[2],z[3],z[4]))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033',sig.index.year==2033)]:
 z=calc(fwd,mask); print(name,'dates',z[0],'IC %.6f ICIR %.6f hit %.4f'%(z[1],z[2],z[3]) if z[0]>1 else 'insufficient')
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).div(len(U)).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340105_volume_pressure_reversal_signal.csv',index=False)
