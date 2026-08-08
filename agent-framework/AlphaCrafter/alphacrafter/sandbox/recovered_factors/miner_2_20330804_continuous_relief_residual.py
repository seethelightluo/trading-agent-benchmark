import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
p=pd.DataFrame({a:load(a) for a in A}).sort_index(); r=np.log(p).diff()
def mac(a):
 x=pd.read_csv('../persistent/index_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float)
 return np.log(x).diff().reindex(p.index).ffill()
v,d=mac('VIX'),mac('DXY'); bench=r.mean(1); resid=r.sub(bench,axis=0)
base=resid.rolling(40,min_periods=30).sum()
# continuous relief intensity: positive when VIX/DXY have fallen, bounded and non-sparse
intensity=(np.tanh(-8*v.rolling(5).sum())+np.tanh(-8*d.rolling(5).sum()))/2
sig=base.mul(intensity,axis=0).shift(1)
print('range',p.index.min(),p.index.max(),'assets',len(A),'coverage',sig.notna().sum().sum()/sig.size,'intensity_positive',np.mean(intensity>0))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8 and sig.loc[dt,ok].nunique()>=3:
   q=spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic
   if np.isfinite(q):z.append(q);ds.append(dt);ns.append(ok.sum())
 z=np.array(z); print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(np.array(ds)>=pd.Timestamp(lo+'-01-01'))&(np.array(ds)<=pd.Timestamp(hi+'-12-31'))]; print(lo+'-'+hi,len(q),('%.6f %.6f'%(q.mean(),q.mean()/q.std(ddof=1))) if len(q) else '')
print('turn10',sig.rank(axis=1,pct=True).diff(10).abs().mean(1).dropna().mean())
