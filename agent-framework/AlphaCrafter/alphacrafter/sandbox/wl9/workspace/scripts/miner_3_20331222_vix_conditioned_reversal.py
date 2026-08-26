import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill();r=cl.pct_change();
v=get_index_daily_data('VIX',days=5000).set_index('date')['close'].reindex(cl.index).ffill(); d=get_index_daily_data('DXY',days=5000).set_index('date')['close'].reindex(cl.index).ffill()
# Contrarian price shock is selectively enabled by broad risk stress: VIX and DXY
# jointly rising over 10 sessions. Lag all inputs one session.
stress=((v.pct_change(10)>0)&(d.pct_change(10)>0)).astype(float)
sig=(-cl.pct_change(20)/(r.rolling(20).std()*np.sqrt(252)+.04)*(0.65+0.7*stress.values[:,None])).clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 f=cl.shift(-h)/cl-1;xs=[];ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q):xs.append(q);ns.append(ok.sum())
 x=pd.Series(xs);print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)))
f=cl.shift(-60)/cl-1
for n,m in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 x=[]
 for dt in sig.index[m]:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q):x.append(q)
 x=pd.Series(x);print(n,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_3_20331222_vix_conditioned_reversal_signal.csv',index=False)
