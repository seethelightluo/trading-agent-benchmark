import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in A},axis=1).sort_index(); r=p.pct_change(); end=pd.Timestamp('2031-03-05'); p=p.loc[:end];r=r.loc[:end]
# downside-resilient momentum: 20d return divided by downside deviation over 60d
neg=r.where(r<0); dv=neg.rolling(60,min_periods=40).std(); f=r.rolling(20,min_periods=20).sum()/dv
for h in [5,10,20]:
 fut=p.shift(-h).div(p)-1; ic=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ic); print(h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1)*np.sqrt(252),4),round((a>0).mean(),4),round(np.mean(ns)/15,4))
print('turnover10',round(f.rank(pct=True).diff(10).abs().stack().mean(),6))
