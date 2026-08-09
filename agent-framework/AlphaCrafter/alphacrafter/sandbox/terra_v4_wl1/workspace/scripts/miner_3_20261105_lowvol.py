import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change()
# residual low volatility: volatility relative to contemporaneous cross-section, rewarding stable assets
v=r.rolling(30,min_periods=20).std(); f=-v
for label,fac in [('lowvol',f),('shock_resilience',r.rolling(20,min_periods=15).quantile(.2))]:
 for h in [1,5,10]:
  q=[]; ns=[]
  yy=p.pct_change(h).shift(-h)
  for i in range(30,len(p)-h):
   z=pd.concat([fac.iloc[i],yy.iloc[i]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  q=np.array(q); print(label,h,'dates',len(q),'names',np.mean(ns),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1),'hit',np.mean(q>0))
