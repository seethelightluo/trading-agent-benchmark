import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={};
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x
# volume-confirmed intermediate momentum: trailing 10d return weighted by current volume surprise
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); idx=pd.DatetimeIndex(idx)
f=pd.DataFrame(index=idx); ret=pd.DataFrame(index=idx)
for s,x in D.items():
 c=x.close.reindex(idx); v=x.volume.replace(0,np.nan).reindex(idx)
 ret[s]=c.pct_change(); vs=np.log(v/v.rolling(20,min_periods=15).median())
 f[s]=c.pct_change(10)*vs
f=f.loc[:'2026-07-15']; ret=ret.loc[f.index]
for h in [1,5,10]:
 y=ret[U].shift(-1).rolling(h).sum(); a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(a);print('h',h,'N',len(a),'names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),4))
# regimes daily
for yr in range(2020,2027):
 q=a[[x.year==yr for x in ds]] if h==1 else None
# rank turnover daily
print('turnover',f.rank(pct=True).diff().abs().mean().mean())
