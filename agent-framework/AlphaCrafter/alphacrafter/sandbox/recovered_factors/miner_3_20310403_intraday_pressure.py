import pandas as pd,numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[a]=d
idx=sorted(set().union(*[set(x.index) for x in D.values()])); idx=pd.DatetimeIndex(idx)
def col(k): return pd.DataFrame({a:D[a][k] for a in ASSETS},index=idx)
o,h,l,c=col('open'),col('high'),col('low'),col('close')
r=c.pct_change()
# Persistent intraday buying pressure: close location / range, smoothed and volatility scaled.
cl=((c-o)/(h-l).replace(0,np.nan)).rolling(10,min_periods=7).mean()
sig=cl/(r.rolling(20,min_periods=15).std()*np.sqrt(20))
fwd={h:c.shift(-h)/c-1 for h in [1,5,10,20]}
for H in fwd:
 v=[]; ns=[]
 for dt in idx:
  z=pd.concat([sig.loc[dt],fwd[H].loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(v);print('H',H,'dates',len(a),'meanN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('coverage',sig.notna().mean().mean(),'cells',sig.notna().sum().sum(),'turn10',sig.rank(axis=1,pct=True).diff(10).abs().mean().mean())
for name,mask in [('2020-23',(idx.year<=2023)),('2024-27',((idx.year>=2024)&(idx.year<=2027))),('2028-30',((idx.year>=2028)&(idx.year<=2030))),('latest120',np.arange(len(idx))>=len(idx)-120)]:
 v=[]
 for dt in idx[mask]:
  z=pd.concat([sig.loc[dt],fwd[10].loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(v);print(name,len(a),np.nanmean(a) if len(a) else np.nan,np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
