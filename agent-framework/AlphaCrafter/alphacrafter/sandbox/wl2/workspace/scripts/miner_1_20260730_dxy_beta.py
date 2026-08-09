import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 d=pd.read_csv('../persistent/stock_data/'+sym+'.csv',parse_dates=['date']).set_index('date'); return d.close.pct_change()
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.pct_change().rename('m')
r=pd.concat([load(x).rename(x) for x in U]+[macro],axis=1).sort_index()
# defensive DXY beta: lower/negative beta ranks higher
m=r.m
fac=pd.DataFrame(index=r.index)
for x in U:
 cov=r[x].rolling(60,min_periods=45).cov(m); var=m.rolling(60,min_periods=45).var()
 fac[x]=-cov/var
# forward next-day return
fr=r[U].shift(-1)
ics=[]; turnovers=[]; counts=[]
for dt in fac.index:
 a=fac.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); counts.append(len(z))
# rank turnover adjacent valid dates
rank=fac.rank(axis=1,pct=True); turnovers=rank.diff().abs().mean(axis=1).dropna()
a=np.array(ics); print('dates',len(a),'meanN',np.mean(counts),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
print('coverage',fac.stack().notna().mean(),'turnover',turnovers.mean())
for y in [2020,2021,2022,2023,2024,2025,2026]:
 vals=[]
 for dt in fac.index:
  if dt.year==y:
   z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(y,len(vals),round(np.mean(vals),5) if vals else None,round(np.mean(vals)/np.std(vals,ddof=1),4) if len(vals)>1 else None)
