import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in assets}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change(); vv=v.reindex(p.index).ffill(); vm=vv.rolling(60,min_periods=30).median(); scale=(vm/vv).clip(.25,4)
f=r.rolling(20,min_periods=20).sum().mul(scale,axis=0)
end=pd.Timestamp('2031-03-05'); f=f.loc[:end]
for h in [5,10,20]:
  fut=p.shift(-h).div(p)-1; vals=[]; ns=[]
  for dt in f.index:
    z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  a=np.array(vals); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR_ann',round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(252),4),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
rank=f.rank(axis=1,pct=True); print('turnover10',round(rank.diff(10).abs().stack().mean(),6),'end',end.date())
