import numpy as np,pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-02')
def load(s):
 for root in ['../persistent/stock_data','../persistent/index_data']:
  p=os.path.join(root,s+'.csv')
  if os.path.exists(p):
   d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]['close'].astype(float)
 return None
C=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index()
r=C.pct_change(); vol=r.rolling(20,min_periods=15).std();
# Volatility-normalized short reversal, damped when recent volatility is rising.
shock=r.rolling(5,min_periods=5).sum(); state=vol/vol.rolling(60,min_periods=30).median()
sig=(-shock/(vol*np.sqrt(5)+1e-12) * (1/(1+state))).shift(1)
print('assets',len(C.columns),'cutoff',C.index.max(),'dates',len(C))
for h in [1,5,10,20]:
 f=C.shift(-h)/C-1; vals=[]; ns=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a))
 x=pd.Series(vals); print('h',h,'dates',len(x),'avg_n',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',(x>0).mean())
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
   # reconstruct date aligned
   zz=pd.Series(vals,index=[d for d in sig.index if len(pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna())>=8])
   z=zz[(zz.index.year>=lo)&(zz.index.year<=hi)]; print('regime',lo,hi,len(z),z.mean())
# coverage and turnover
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
sig.stack().rename('signal').to_csv('scripts/miner_3_20270303_vol_damped_reversal_signal.csv')
