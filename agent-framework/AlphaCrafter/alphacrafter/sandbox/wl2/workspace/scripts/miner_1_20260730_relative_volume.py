import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def get(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()[c].loc[:cut]
V=pd.concat({a:get(a,'volume') for a in U},axis=1).sort_index(); R=pd.concat({a:get(a,'close').pct_change() for a in U},axis=1).reindex(V.index)
# Relative volume acceleration: short/medium volume ratio, cross-sectionally rankable.
F=np.log((V.rolling(5,min_periods=5).mean())/(V.rolling(40,min_periods=30).mean()))
for h in [1,5,10]:
 Y=R.rolling(h,min_periods=h).sum().shift(-h+1); x=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],Y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.asarray(x);print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
print('coverage',F.stack().notna().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for yr in range(2020,2027):
 x=[]
 for d in F.index[F.index.year==yr]:
  z=pd.concat([F.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(yr,len(x),round(np.mean(x),5) if x else None,round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
