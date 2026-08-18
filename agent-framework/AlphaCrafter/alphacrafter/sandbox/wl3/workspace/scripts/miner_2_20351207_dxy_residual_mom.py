import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close']
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change(); dr=dxy.pct_change()
# DXY-residualized risk-adjusted momentum: 20d return/20d vol, less rolling beta*DXY momentum.
rollcov=ret.rolling(60).cov(dr); vard=dr.rolling(60).var()
beta=rollcov.div(vard,axis=0).clip(-3,3)
raw=prices.pct_change(20)/ret.rolling(20).std().replace(0,np.nan)
sig=raw-beta.multiply(dr.rolling(20).sum(),axis=0)
# signal observed t predicts t+1..t+10; lag one day
f=sig.shift(1); fw=prices.shift(-10)/prices-1
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fw.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate DXY-residualized risk-adjusted momentum20; dates',len(r),'assets',len(assets),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(),'hit',(r.ic>0).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2030-12-31'),('2031','2035-12-31')]:
 q=r.loc[lo:hi].ic; print(lo,hi,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,3,5,10,20]:
 fw=prices.shift(-h)/prices-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
# rank turnover
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
# artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20351207_dxy_residual_mom_signal.csv',index=False)
