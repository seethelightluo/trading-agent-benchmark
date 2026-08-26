import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
disp=r.rolling(5,min_periods=4).std().mean(axis=1); q=disp.rolling(252,min_periods=100).rank(pct=True)
active=(q>=.45)&(q<=.70); base=-r.rolling(10,min_periods=8).sum(); f=base.shift(1); y=p.pct_change(10).shift(-10)
A=[];D=[];ns=[]; turns=[]
for d in f.index[active.fillna(False)]:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);D.append(d);ns.append(len(z)); turns.append(np.mean(f.loc[d].rank()!=f.shift(1).loc[d].rank()))
A=np.asarray(A); D=pd.DatetimeIndex(D)
print('factor=dispersion_gated_reversal10 active-date IC');print('dates',len(A),'avgN',round(np.mean(ns),2),'active_fraction',round(active.mean(),4),'coverage',round(np.mean([len(pd.concat([f.loc[d],y.loc[d]],axis=1).dropna())/15 for d in D]),4),'turnover',round(np.mean(turns),4),'IC',round(A.mean(),6),'ICIR',round(A.mean()/A.std(ddof=1),6),'hit',round(np.mean(A>0),4))
for name,lo,hi in [('2020_22','2020','2022-12-31'),('2023_25','2023','2025-12-31'),('2026_28','2026','2028-12-31'),('2029','2029','2029-11-05')]:
 z=A[(D>=lo)&(D<=hi)];print(name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
f.where(active, np.nan).to_csv('scripts/miner_2_20291105_dispersion_gated_reversal10_signal.csv',index_label='date')
