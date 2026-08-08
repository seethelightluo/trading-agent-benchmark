import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=p.pct_change()
# Risk-adjusted medium trend: 40d return divided by 20d realized volatility; interpretable trend quality
f=r.rolling(40,min_periods=30).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20))
print('candidate risk_adjusted_40d_trend')
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; ic=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(d)
 a=np.array(ic); print('H',h,'dates',len(a),'meanIC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'meanN',np.mean(ns))
for label,sl in [('2024-27',slice('2024','2027')),('2028-30',slice('2028','2030')),('2031+',slice('2031',None)),('recent120',slice(None,None))]:
 q=(p.shift(-1)/p-1); x=[]
 for d in f.loc[sl].index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x[-120:] if label=='recent120' else x); print(label,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
# dates available
print('date range',f.index.min(),f.index.max())
