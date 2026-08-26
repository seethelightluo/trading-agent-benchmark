import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=cl.pct_change(); v= r.rolling(60).std()*np.sqrt(252)
r20=cl.pct_change(20); r60=cl.pct_change(60); r120=cl.pct_change(120)
# Agreement-weighted risk-adjusted momentum: mean horizon returns normalized by 60D risk,
# with a multiplier rewarding consistent signs across 20/60/120D horizons.
raw=(.25*r20+.45*r60+.30*r120)/(v+.05)
agree=(np.sign(r20)+np.sign(r60)+np.sign(r120)).abs()/3
sig=(raw*(.5+.5*agree)).shift(1).clip(-5,5)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 fwd=cl.shift(-h)/cl-1; xs=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q); ns.append(ok.sum())
 x=pd.Series(xs); print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[mask]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q)
 x=pd.Series(xs); print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).div(len(U)).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20331208_multihorizon_agreement_momentum_signal.csv',index=False)
