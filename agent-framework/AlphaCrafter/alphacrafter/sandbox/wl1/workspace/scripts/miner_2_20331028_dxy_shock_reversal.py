import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv');q.date=pd.to_datetime(q.date);P[s]=q.sort_values('date').set_index('date').close.astype(float)
pdxy=pd.read_csv('../persistent/index_data/DXY.csv');pdxy.date=pd.to_datetime(pdxy.date);dxy=pdxy.sort_values('date').set_index('date').close.astype(float)
p=pd.concat(P,axis=1).sort_index().ffill(); dxy=dxy.reindex(p.index).ffill(); r=np.log(p).diff()
# DXY-shock-conditioned cross-asset reversal: stronger reversal after dollar rallies,
# while retaining only cross-sectional relative 10d losses and scaling by idiosyncratic volatility.
dshock=np.log(dxy).diff(5).rolling(60,min_periods=30).rank(pct=True)
condition=((dshock-.50)*2).clip(-1,1)
vol=r.rolling(30,min_periods=20).std()
f=(-np.log(p).diff(10)/(vol*np.sqrt(10)+1e-8)).mul(condition,axis=0).shift(1)
y=np.log(p).shift(-10)-np.log(p); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic):rows.append((dt,ic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-01':'2033-10-26']
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=x.loc[a:b];print(a,b,len(q), 'IC %.6f ICIR %.6f hit %.3f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean();print('turnover',turn)
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20331028_dxy_shock_reversal_signal.csv',index=False)
x.to_csv('scripts/miner_2_20331028_dxy_shock_reversal_ic.csv')
