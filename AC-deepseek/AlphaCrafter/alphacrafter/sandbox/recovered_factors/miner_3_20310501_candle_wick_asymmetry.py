import pandas as pd,numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in ASSETS}
idx=pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in D.values()])))
def col(k): return pd.DataFrame({a:D[a][k] for a in ASSETS},index=idx)
o,h,l,c=col('open'),col('high'),col('low'),col('close')
# Candle asymmetry: persistent lower-wick demand minus upper-wick supply,
# normalized by true range and averaged over 10 sessions, then residualized by market median.
tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=0).groupby(level=0).max()
upper=(h-np.maximum(o,c))/tr.replace(0,np.nan)
lower=(np.minimum(o,c)-l)/tr.replace(0,np.nan)
raw=(lower-upper).rolling(10,min_periods=7).mean()
sig=raw.sub(raw.median(axis=1),axis=0)
for H in [1,5,10,20]:
 f=c.shift(-H)/c-1; vals=[];ns=[]
 for dt in idx:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',H,'dates',len(a),'meanN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
print('coverage_cells',sig.notna().sum().sum(),'total',sig.size,'coverage',sig.notna().mean().mean(),'turn10',sig.rank(axis=1,pct=True).diff(10).abs().mean().mean())
f=c.shift(-10)/c-1
for name,mask in [('2020-23',idx.year<=2023),('2024-27',(idx.year>=2024)&(idx.year<=2027)),('2028-30',(idx.year>=2028)&(idx.year<=2030)),('latest120',np.arange(len(idx))>=len(idx)-120)]:
 v=[]
 for dt in idx[mask]:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(v);print(name,'dates',len(a),'IC',np.mean(a) if len(a) else np.nan,'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
