import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-09-09')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change(); common=r.mean(axis=1); resid=r.sub(common,axis=0)
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
for label,w,smooth in [('3d',3,3),('5d',5,2),('10d',10,2)]:
 rr=resid.rolling(w,min_periods=w).sum().shift(1); rv=resid.rolling(20,min_periods=20).std().shift(1); f=(-rr/rv).rolling(smooth,min_periods=smooth).mean(); rows=[]
 for i,d in enumerate(p.index[:-20]):
  if d<pd.Timestamp('2020-04-01') or p.index[i+10]>cut:continue
  q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
  if pd.notna(q):rows.append(q)
 z=pd.Series(rows); print(label,'dates',len(z),'coverage',f.notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turn',f.rank(pct=True).diff().abs().mean().mean())
 print('recent365',z.tail(365).mean(),z.tail(365).mean()/z.tail(365).std(ddof=1))
# persist artifacts for chosen 5d only for audit
f=(-resid.rolling(5,min_periods=5).sum().shift(1)/resid.rolling(20,min_periods=20).std().shift(1)).rolling(2,min_periods=2).mean()
f.to_csv('scripts/miner_3_20320916_residual_reversal_5d_signal.csv')
