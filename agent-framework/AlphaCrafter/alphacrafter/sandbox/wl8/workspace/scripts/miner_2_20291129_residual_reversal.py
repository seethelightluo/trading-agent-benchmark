import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in S},axis=1).sort_index().loc[:'2029-11-28']
r=P.pct_change(); bench=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0)
res=r.rolling(5,min_periods=5).sum()-beta.mul(bench.rolling(5,min_periods=5).sum(),axis=0)
vol=r.rolling(20,min_periods=15).std()
# Contrarian residual move, risk normalized; all estimates lagged one day.
sig=(-res/vol.replace(0,np.nan)).shift(1)
rows=[]
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; vals=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(vals,columns=['date','n','ic']).set_index('date')
 print('h',h,'dates',len(a),'avg_n',round(a.n.mean(),3),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 if h==10:
  for lo,hi in [('2020','2023'),('2024','2026-07-15'),('2026-07-16','2027-12-31'),('2028','2029-11-28')]:
   x=a.loc[lo:hi].ic; print('regime',lo,hi,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else np.nan)
  out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20291129_residual_reversal_signal.csv',index=False)
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
