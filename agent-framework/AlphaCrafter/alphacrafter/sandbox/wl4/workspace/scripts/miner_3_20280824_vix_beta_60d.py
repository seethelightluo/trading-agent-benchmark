import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data';M='../persistent/index_data'
pr=pd.DataFrame({s:pd.read_csv(os.path.join(P,s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=pr.pct_change()
v=pd.read_csv(os.path.join(M,'VIX.csv'),parse_dates=['date']).set_index('date')['close'].sort_index().pct_change().reindex(pr.index).fillna(0)
cov=r.rolling(60,min_periods=45).cov(v); var=v.rolling(60,min_periods=45).var()
# low VIX beta is defensive; lag prevents lookahead
f=(-cov.div(var,axis=0)).shift(1)
def one(dt,h):
 y=(1+r.loc[dt:].iloc[1:h+1]).prod()-1; z=pd.concat([f.loc[dt],y],axis=1).dropna()
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
