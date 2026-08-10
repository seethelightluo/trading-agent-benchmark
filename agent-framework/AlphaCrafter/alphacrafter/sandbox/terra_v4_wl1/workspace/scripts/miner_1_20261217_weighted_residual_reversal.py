import numpy as np,pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
b=Path('../persistent/stock_data'); P=pd.DataFrame({s:pd.read_csv(b/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:END]
R=P.pct_change(); med=R.median(axis=1); resid=R.sub(med,axis=0)
# lagged, volatility-controlled short-horizon residual reversal
raw=(0.2*resid.rolling(1).sum()+0.5*resid.rolling(3).sum()+0.3*resid.rolling(5).sum())
vol=R.rolling(20,min_periods=15).std(); F=(-raw/vol).shift(1)
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; vals=[]; ns=[]; dates=[]
 for d in P.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(z.f.corr(z.y,method='spearman'));ns.append(len(z));dates.append(d)
 x=pd.Series(vals,index=dates); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
   q=x.loc[lo:hi]; print('REG',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20261217_weighted_residual_reversal_signal.csv',index=False)
