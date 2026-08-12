import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-06-25')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:end]
R=P.pct_change(); vol=R.rolling(20).std(); raw=P.pct_change(20); resid=raw.sub(raw.median(axis=1),axis=0)
# volatility-scaled residual momentum, active only in subdued VIX regime
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill()
quiet=vix < vix.rolling(120,min_periods=60).median()
f=resid/vol; f=f.where(quiet)
print('active dates',quiet.sum(),'total',len(quiet),'assets',len(U))
for h in [3,5,10]:
 a=[]; dates=[]; ns=[]
 for i,d in enumerate(P.index):
  if i+h>=len(P): continue
  z=pd.concat([f.loc[d].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1:
   a.append(spearmanr(z.x,z.y).statistic); dates.append(d); ns.append(len(z))
 a=np.asarray(a); print('h',h,'dates',len(a),'avgN',np.mean(ns),'coverage_dates',len(a)/max(1,quiet.sum()),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
# rank turnover on active consecutive dates
print('coverage',f.notna().sum().sum()/(len(P)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_1_20310626_quiet_residual_momentum_signal.csv',index_label='date')
