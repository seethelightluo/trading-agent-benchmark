import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:'2027-02-24']
r=P.pct_change(); sig=r.rolling(20).sum()-0.5*r.rolling(5).sum()
for h in [1,3,5,10]:
 f=P.shift(-h)/P-1; ics=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ics); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
f=P.shift(-5)/P-1; a=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(a,columns=['date','ic']).set_index('date')
for period,g in a.groupby(a.index.year//2*2):print('regime',period,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),5))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'turnover',round(sig.rank(pct=True).diff().abs().mean().mean(),4))
