import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<100:d=get_index_daily_data(s,2600)
 if d is not None and len(d): D[s]=d.set_index(pd.to_datetime(d.date)).sort_index()['close']
p=pd.DataFrame(D); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std();
# low-volatility trend: 60d return / 60d volatility, lagged
f=(p.pct_change(60)/(vol.rolling(60,min_periods=40).mean()*np.sqrt(60))).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); out=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(out).dropna(); print('h',h,'dates',len(x),'names',np.mean(ns) if ns else 0,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan,'hit',(x>0).mean() if len(x) else np.nan)
print('range',p.index.min(),p.index.max(),'coverage',f.notna().mean().mean())
