import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Candidate: volatility-managed cross-sectional defensive factor, inverse 20d realized vol, with 60d volatility trend penalty
rv=r.rolling(20,min_periods=15).std(); longrv=r.rolling(60,min_periods=40).std()
f=-(rv/longrv).replace([np.inf,-np.inf],np.nan) # lower recent vol relative to long vol
# evaluate daily cross-sectional IC, forward horizons
for h in [1,5,10]:
  vals=[]
  for i in range(len(P)-h):
    # signal at t, return t+1...t+h, strictly completed data
    a=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
    z=pd.concat([a,y],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=np.array(vals); print(h,len(x), 'avgN', np.nan, 'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0))
# regimes and turnover
x=f.rank(axis=1,pct=True); to=(x.diff().abs().mean(axis=1)).mean(); print('coverage',f.notna().mean().mean(),'turnover',to)
# artifact all dates signals
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_2_20270225_relvol_ratio.csv',index=False)
