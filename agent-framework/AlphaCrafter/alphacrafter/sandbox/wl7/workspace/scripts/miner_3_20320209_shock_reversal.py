import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U};P=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); lr=np.log(P).diff()
# A short shock-reversal signal: fade the latest 1-day move, but emphasize moves that are
# unusual versus each asset's lagged 20-day volatility; cross-sectional demean removes common shocks.
z=lr.div(lr.rolling(20,min_periods=15).std()).shift(1); fac=-z.sub(z.median(axis=1),axis=0)
for h in [1,3,5,10]:
 y=P.pct_change(h).shift(-h);a=[];n=[];ds=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c);n.append(len(q));ds.append(dt)
 a=np.array(a);print('H',h,'dates',len(a),'avgN',round(np.mean(n),3),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),5))
 if h==5:print('regimes',*[round(x.mean(),8) for x in np.array_split(a,3)])
print('history_dates',len(P),'assets',len(P.columns),'coverage',round(fac.notna().mean().mean(),6),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
fac.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20320209_shock_reversal_signal.csv',index=False)
