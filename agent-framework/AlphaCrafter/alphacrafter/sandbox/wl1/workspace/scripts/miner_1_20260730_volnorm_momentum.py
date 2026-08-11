import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index();r=p.pct_change()
mom=p/p.shift(10)-1; vol=r.rolling(20,min_periods=10).std()*np.sqrt(252); f=mom/vol.replace(0,np.nan)
# explicit row-wise winsorization avoids cross-sectional alignment issues
f=f.apply(lambda x: x.clip(x.dropna().quantile(.05),x.dropna().quantile(.95)) if x.notna().any() else x,axis=1)
for h in [1,5,10]:
 ic=[];ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q))
 x=np.array(ic);print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
print('assets',len(U),'valid_dates',f.notna().any(axis=1).sum(),'turnover_proxy',round((f.rank(pct=True,axis=1).diff().abs().mean(axis=1)>.25).mean(),4))
