import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); d[s]=x.close.astype(float)
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# intermediate-horizon risk-adjusted momentum, lagged
f=(p.pct_change(40)/r.rolling(40,min_periods=30).std()).shift(1)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(q)>=8: a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(a);print(h,len(a),round(np.mean(ns),2),round(np.nanmean(a),5),round(np.nanmean(a)/np.nanstd(a,ddof=1),5),round(np.mean(a>0),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),5))
