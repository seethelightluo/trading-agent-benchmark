import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data'
dfs={s:pd.read_csv(os.path.join(P,s+'.csv'),parse_dates=['date']).set_index('date').sort_index() for s in U}
pr=pd.DataFrame({s:x.close for s,x in dfs.items()}); r=pr.pct_change(); vol=pd.DataFrame({s:x.volume for s,x in dfs.items()})
# momentum confirmed by unusual volume, all lagged one day
ret=r.rolling(5,min_periods=5).sum(); vs=np.log1p(vol).sub(np.log1p(vol).rolling(60,min_periods=30).mean()); f=(ret*vs).shift(1)
def one(dt,h):
 y=(1+r.loc[dt:].iloc[1:h+1]).prod()-1;z=pd.concat([f.loc[dt],y],axis=1).dropna()
 if len(z)<8:return np.nan,len(z)
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)
for h in [1,5,10,20]:
 a=[];n=[]
 for dt in pr.index:
  try:x,k=one(dt,h)
  except:continue
  if np.isfinite(x):a.append(x);n.append(k)
 a=pd.Series(a);print('H',h,'dates',len(a),'avgN',round(np.mean(n),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028-08-23')]:
 z=[]
 for dt in pr.loc[a:b].index:
  try:x,k=one(dt,10)
  except:continue
  if np.isfinite(x):z.append(x)
 z=pd.Series(z);print('REG',a,'dates',len(z),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5))
