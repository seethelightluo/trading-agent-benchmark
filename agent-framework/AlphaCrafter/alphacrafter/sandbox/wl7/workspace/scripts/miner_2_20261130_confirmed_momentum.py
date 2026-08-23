import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame(p).sort_index().loc[:'2026-11-30']; r=p.pct_change()
# lagged 10-day momentum with a 3-day confirmation, simple and interpretable
f=(p.shift(1)/p.shift(11)-1) * (1+0.5*np.sign(p.shift(1)/p.shift(4)-1))
for h in [1,5,10]:
 y=p.shift(-h)/p-1;a=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(a);print('h',h,'dates',len(a),'avg_names',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean(np.array(ns)/15))
print('turnover',np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)))
