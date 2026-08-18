import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,4500)
 if d is None or len(d)==0:d=get_index_daily_data(s,4500)
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame({s:L(s) for s in A}).sort_index(); r=p.pct_change()
# Residual reversal: remove daily common cross-asset component, then fade accumulated 20d residual losses.
cs=r.mean(axis=1); resid=r.sub(cs,axis=0)
res20=resid.rolling(20,min_periods=12).sum(); rv=resid.rolling(60,min_periods=30).std()
f=(-res20/(rv*np.sqrt(20)+1e-8)).shift(1)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 s=pd.Series(z).dropna();print('horizon',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
valid=f.notna().sum(axis=1);use=valid>=8
print('dates',int(use.sum()),'assets',len(A),'coverage',round(valid[use].mean()/15,4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).where(use).mean()),6))
f.loc[use].to_csv('../persistent/miner_2_20341124_residual_reversal_signal.csv',index_label='date')
