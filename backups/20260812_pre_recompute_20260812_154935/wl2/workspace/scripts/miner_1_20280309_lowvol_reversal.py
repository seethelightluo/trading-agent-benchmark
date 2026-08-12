import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3200)
 if d is not None and len(d):
  d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=np.log(P).diff(); Y=np.log(P).shift(-1)-np.log(P)
# Volatility-managed reversal: low recent volatility is preferred, with a mild short-term
# reversal component; both components are lagged and cross-sectionally standardized.
vol=r.rolling(20,min_periods=20).std(); rev=-r.rolling(5,min_periods=5).sum();
def csz(x):return x.sub(x.mean(axis=1),axis=0).div(x.std(axis=1).replace(0,np.nan),axis=0)
F=(csz(-vol)+0.35*csz(rev)).shift(1)
def calc(y,sl=slice(None)):
 a=[];n=[]
 for dt in F.loc[sl].index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z))
 x=pd.Series(a).dropna();return len(x),np.mean(n),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('range',P.index.min(),P.index.max(),'symbols',len(P.columns))
for h in [1,3,5,10]:print('h',h,calc(np.log(P).shift(-h)-np.log(P)))
print('coverage',F.notna().sum(axis=1).mean()/15,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028',slice('2028',None))]:print(name,calc(Y,sl))
