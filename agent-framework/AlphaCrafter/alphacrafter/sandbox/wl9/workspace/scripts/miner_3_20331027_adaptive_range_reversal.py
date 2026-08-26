import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=cl.pct_change(); r60=cl.pct_change(60); vol=ret.rolling(60).std()*np.sqrt(252)
# Longer regime-aware range: emphasize medium-term losers near the bottom of 180d range.
lo=cl.rolling(180).min(); hi=cl.rolling(180).max(); pos=(cl-lo)/(hi-lo).replace(0,np.nan)
sig=((-r60/(vol+0.05))*(1.0+0.75*(0.5-pos))).clip(-5,5).shift(1)
rows=[]
for h in [10,20,40,60]:
 fwd=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(b[ok],method='spearman')); ns.append(ok.sum())
 x=pd.Series(vals).dropna(); rows.append((h,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for z in rows: print('H',z[0],'dates',z[1],'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%z[2:])
cov=sig.notna().sum(axis=1)/len(U); turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(); print('coverage %.6f turnover %.6f'%(cov.mean(),turn))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[mask]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: xs.append(sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman'))
 x=pd.Series(xs).dropna(); print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331027_adaptive_range_reversal_signal.csv',index=False)
