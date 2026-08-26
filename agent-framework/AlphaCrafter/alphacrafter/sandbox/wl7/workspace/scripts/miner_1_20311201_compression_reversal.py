import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}; P=pd.DataFrame({s:d.set_index('date').close for s,d in D.items() if d is not None}).sort_index().ffill(); R=np.log(P).diff()
# Reversal is amplified when current volatility is compressed versus its slow baseline; lag avoids look-ahead.
v20=R.rolling(20,min_periods=15).std(); v80=R.rolling(80,min_periods=60).std(); compression=(v80/v20).clip(0.25,4)
f=(-P.pct_change(10)*compression).shift(1)
for h in [1,5,10,20]:
 y=P.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): a.append(q); ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: pd.DataFrame({'ic':a,'n':ns}).to_csv('scripts/miner_1_20311201_compression_reversal_ic.csv',index=False)
print('span',P.index.min(),P.index.max(),'assets',len(P.columns),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
f.to_csv('scripts/miner_1_20311201_compression_reversal_signal.csv')
