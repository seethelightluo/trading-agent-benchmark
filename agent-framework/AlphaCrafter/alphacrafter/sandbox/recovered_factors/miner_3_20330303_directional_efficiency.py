import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in W}).sort_index().loc[:'2033-02-16']
r=p.pct_change()
# Directional path efficiency: net 10-day return divided by total absolute daily movement.
# Lagged one day to respect decision-time observability.
sig=(r.rolling(10,min_periods=8).sum()/(r.abs().rolling(10,min_periods=8).sum()+1e-12)).shift(1)
print('candidate directional_efficiency_10d; dates',len(p),'assets',len(W))
print('coverage',round(sig.notna().mean().mean(),4),'mean_valid',round(sig.notna().sum(axis=1).replace(0,np.nan).mean(),2),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for h in [1,5,10,20]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,lo,hi in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-02-16')]:
 fw=p.pct_change(1).shift(-1); vals=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('REG1',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else 'NA','hit',round((a>0).mean(),4))
print('AUDIT_REQUIRED: candidate failed unless a same-horizon pair meets both gates; exact library correlation audit required before persistence.')
