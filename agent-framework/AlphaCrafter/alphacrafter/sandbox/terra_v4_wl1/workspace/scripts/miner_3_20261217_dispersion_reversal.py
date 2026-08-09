import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=d.close.astype(float).pct_change()
R=pd.concat(D,axis=1,sort=True).loc[:END]
r5=R.rolling(5,min_periods=5).sum(); vol=R.rolling(20,min_periods=15).std(); disp=R.rolling(5,min_periods=5).std().mean(axis=1)
reg=(disp.shift(1)>disp.shift(1).rolling(60,min_periods=30).median()).astype(float)
F=(-(r5.shift(1).sub(r5.shift(1).median(axis=1),axis=0)).div(vol.shift(1))).mul(reg,axis=0)
for h in [1,5,10]:
 y=R.shift(-h).rolling(h).sum().shift(-(h-1)); a=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: a.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
 a=pd.DataFrame(a,columns=['date','ic','n']); q=a.ic
 print('H',h,'dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for yr,g in a.groupby(a.date.dt.year): print('YR',yr,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
print('active',reg.mean(),'coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
F.to_csv('scripts/miner_3_20261217_dispersion_reversal_signal.csv',index_label='date')
print('period',R.index.min(),R.index.max(),'symbols',len(U))
