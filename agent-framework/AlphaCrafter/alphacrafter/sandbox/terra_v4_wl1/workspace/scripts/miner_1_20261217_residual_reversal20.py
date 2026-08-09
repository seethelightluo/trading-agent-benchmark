import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data'); cutoff=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cutoff]; R=P.pct_change(); m=R.mean(axis=1)
b=R.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0); e=R.sub(b.mul(m,axis=0),axis=0)
F=-e.rolling(20,min_periods=15).sum().div(e.rolling(20,min_periods=15).std()).shift(1); F=F.sub(F.median(axis=1),axis=0)
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20261217_residual_reversal20_signal.csv',index=False)
for h in [1,5,10]:
 a=[];ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(a); print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('period',P.index.min().date(),P.index.max().date(),'coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
