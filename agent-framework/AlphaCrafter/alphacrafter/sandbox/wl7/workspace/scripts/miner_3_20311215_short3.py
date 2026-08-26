import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}; P=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); R=np.log(P).diff()
# Three-day residual reversal versus cross-asset median, normalized by slow 60d volatility, lagged.
ret=P.pct_change(3); peer=ret.median(axis=1); sig=-(ret.sub(peer,axis=0)).div(R.rolling(60,min_periods=45).std()*np.sqrt(252)).shift(1)
for h in [1,3,5,10]:
 y=P.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c);ns.append(len(z));dates.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round(np.mean(a>0),5))
 if h==3:
  q=np.array_split(np.arange(len(a)),3); print('regimes',*[round(a[i].mean(),8) for i in q]); pd.DataFrame({'date':dates,'ic':a,'n':ns}).to_csv('scripts/miner_3_20311215_short3_ic.csv',index=False)
print('history_dates',len(P),'assets',len(P.columns),'coverage',round(sig.notna().mean().mean(),6),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20311215_short3_signal.csv',index=False)
print('span',P.index.min(),P.index.max())
