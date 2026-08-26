import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=cl.pct_change(); r20=cl.pct_change(20); r60=cl.pct_change(60)
v60=r.rolling(60).std()*np.sqrt(252)
confirm=np.sign(r20).replace(0,np.nan)
sig=(-r60/(v60+0.04)*(0.65+0.35*confirm)).clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 fwd=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   q=a[ok].corr(b[ok],method='spearman')
   if pd.notna(q): vals.append(q); ns.append(ok.sum())
 x=pd.Series(vals); print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[mask]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q)
 x=pd.Series(xs); print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).mean()/len(U),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20331222_vol_managed_trend_reversal_signal.csv',index=False)
