import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(path): return pd.read_csv(path,parse_dates=['date']).set_index('date').close.astype(float)
p=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index(); r=np.log(p).diff(); resid=r-r.mean(1).values[:,None]
def macro(a): return np.log(close('../persistent/index_data/'+a+'.csv')).diff().reindex(p.index).ffill()
v,d=macro('VIX'),macro('DXY')
zv=v.rolling(60,min_periods=40).sum()/v.rolling(60,min_periods=40).std(); zd=d.rolling(60,min_periods=40).sum()/d.rolling(60,min_periods=40).std()
relief=(np.tanh(-zv)+np.tanh(-zd))/2
# Reverse direction: assets with weak residual momentum during macro relief are expected to mean-revert
sig=-resid.rolling(40,min_periods=30).sum().mul(relief,axis=0).shift(1)
print('range',p.index.min().date(),p.index.max().date(),'assets',len(A),'cells',int(sig.notna().sum().sum()),'coverage',round(sig.notna().sum().sum()/sig.size,6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8 and sig.loc[dt,ok].nunique()>=3:
   q=spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic
   if np.isfinite(q): z.append(q);ds.append(dt);ns.append(int(ok.sum()))
 z=np.array(z); ds=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(z),'meanN',round(np.mean(ns),3),'IC',round(float(z.mean()),6),'ICIR',round(float(z.mean()/z.std(ddof=1)),6),'hit',round(float((z>0).mean()),4))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(ds>=pd.Timestamp(lo+'-01-01'))&(ds<=pd.Timestamp(hi+'-12-31'))]
  print(' regime',lo+'-'+hi,'n',len(q),'IC/ICIR',('%.6f/%.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'NA'))
print('turn10',round(float(sig.rank(axis=1,pct=True).diff(10).abs().mean(1).dropna().mean()),6))
print('NOTE: library correlation audit required before admission.')
