import pandas as pd,numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in ASSETS}
idx=pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in D.values()])))
c=pd.DataFrame({a:D[a]['close'] for a in ASSETS},index=idx).ffill()
# Recovery efficiency: trailing 120-day drawdown trough rebound, scaled by time spent below peak.
# Positive values favor assets that recover losses rapidly; residualized cross-sectionally.
peak=c.rolling(120,min_periods=60).max()
dd=c/peak-1
# days since each asset's trailing minimum; rebound from min divided by age, with drawdown depth penalty avoided
trough=c.rolling(120,min_periods=60).min()
age=pd.DataFrame(index=idx,columns=ASSETS,dtype=float)
for a in ASSETS:
 x=c[a].values; out=np.full(len(x),np.nan)
 for t in range(len(x)):
  lo=max(0,t-119); j=lo+np.nanargmin(x[lo:t+1]) if t>=lo else lo
  out[t]=t-j+1
 age[a]=out
rebound=c/trough-1
raw=(rebound/np.sqrt(age)).rolling(5,min_periods=3).mean()
# penalize still-deep drawdowns, so rapid recovery from genuine loss scores best
sig=(raw*(1+dd)).sub((raw*(1+dd)).median(axis=1),axis=0)
print('total_dates',len(idx),'assets',len(ASSETS),'cells',sig.notna().sum().sum(),'coverage',sig.notna().mean().mean())
for H in [1,5,10,20]:
 f=c.shift(-H)/c-1; vals=[];ns=[]
 for dt in idx:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',H,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('turn10',sig.rank(axis=1,pct=True).diff(10).abs().mean().mean())
for name,mask in [('2020-23',idx.year<=2023),('2024-27',(idx.year>=2024)&(idx.year<=2027)),('2028-30',(idx.year>=2028)&(idx.year<=2030)),('2031',idx.year==2031),('latest120',np.arange(len(idx))>=len(idx)-120)]:
 v=[]
 for dt in idx[mask]:
  z=pd.concat([sig.loc[dt],(c.shift(-10)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(v);print(name,'dates',len(a),'IC',round(np.mean(a),6) if len(a) else None,'ICIR',round(np.mean(a)/np.std(a,ddof=1),6) if len(a)>1 else None)
