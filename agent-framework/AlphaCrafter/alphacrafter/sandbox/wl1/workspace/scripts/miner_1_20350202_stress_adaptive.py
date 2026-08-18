import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').close for s in U}
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill()
# Observable stress: VIX level relative to trailing 120-day history, lagged one day in signal.
vz=(vix-vix.rolling(120,min_periods=60).mean())/vix.rolling(120,min_periods=60).std()
stress=(vz.clip(-2,2)/2).clip(0,1).shift(1)
# Adaptive blend: trend in calm regimes, short-horizon reversal during elevated stress.
trend=P.pct_change(20).shift(1)
rev=-P.pct_change(5).shift(1)
f=trend*(1-stress.values[:,None])+rev*stress.values[:,None]
# cross-sectional demean avoids common market direction
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for h in [5,10,20,40]:
  ic=[]
  for i,d in enumerate(P.index):
    if i+h>=len(P): break
    x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: ic.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  q=pd.DataFrame(ic,columns=['date','ic','n']).set_index('date')
  print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4))
  if h==10:
   for a,b in [('2020','2024'),('2025','2029'),('2030','2035')]:
    z=q.loc[a:b]; print('REG',a,b,len(z),round(z.ic.mean(),5),round(z.ic.mean()/z.ic.std(ddof=1),5))
# turnover based rank changes every day
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).mean(),5),'valid_dates',int(f.notna().all(axis=1).sum()))
f.to_csv('scripts/miner_1_20350202_stress_adaptive_signal.csv',index_label='date')
print('artifact written')
