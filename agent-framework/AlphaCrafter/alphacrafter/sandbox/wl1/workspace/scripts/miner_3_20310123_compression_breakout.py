import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; close={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):close[s]=d.drop_duplicates('date').set_index('date')['close']
p=pd.DataFrame(close).sort_index().ffill(); r=p.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Breakout after compression: medium trend, boosted when recent volatility is compressed vs its baseline.
compression=(v20/(v60+1e-12)).clip(0.25,2.5)
raw=p.pct_change(20)/(v20*np.sqrt(20)+1e-12)*(1.5-compression)
signal=raw.rank(axis=1,pct=True).shift(1); rows=[]
for i in range(len(p)-21):
 for h in [1,5,10,20]:
  z=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((p.index[i],h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',signal.notna().mean().mean(),'turnover',signal.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
signal.to_csv('scripts/miner_3_20310123_compression_breakout_signal.csv')
