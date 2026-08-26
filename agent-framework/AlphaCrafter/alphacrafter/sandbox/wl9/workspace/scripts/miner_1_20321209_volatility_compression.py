import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2032-12-09')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d.loc[d.index<=cutoff,'close'].replace(0,np.nan)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
fac=(vol60/vol20).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print(h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x)),6),'hit',round(np.mean(x>0),4))
h=20; fr=p.shift(-h)/p-1
for yr in [2024,2025,2026,2027,2028,2029,2030,2031,2032]:
 x=[]
 for dt in fac.index[fac.index.year==yr]:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if x: print('REG',yr,len(x),round(float(np.mean(x)),6))
print('coverage',round(fac.notna().sum().sum()/(fac.shape[0]*fac.shape[1]),4),'turnover',round(fac.rank(pct=True).diff().abs().mean().mean(),6),'end',fac.index[-1].date())
