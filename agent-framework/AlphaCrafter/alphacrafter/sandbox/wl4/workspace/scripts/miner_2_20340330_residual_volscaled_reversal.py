import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): P[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2034-03-29']
R=P.pct_change(); market=R.mean(axis=1)
# Residualized intermediate reversal: remove 60d beta times equal-weight market return.
mu=R.rolling(60,min_periods=35).mean(); mm=market.rolling(60,min_periods=35).mean()
cov=((R.sub(mu,axis=0)).mul(market-mm,axis=0)).rolling(60,min_periods=35).mean()
beta=cov.div(market.rolling(60,min_periods=35).var(),axis=0)
resid60=P.pct_change(60)-beta*market.rolling(60,min_periods=35).sum().values[:,None]
vol=R.rolling(40,min_periods=30).std()*np.sqrt(252)
F=(-resid60.div(vol.replace(0,np.nan))).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(P.columns))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(P.columns),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20340330_residual_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20340330_residual_reversal_signal.csv')
