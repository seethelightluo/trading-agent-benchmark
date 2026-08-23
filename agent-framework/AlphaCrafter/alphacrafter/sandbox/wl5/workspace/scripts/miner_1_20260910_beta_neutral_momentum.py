import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
R=pd.DataFrame(P).pct_change(); m=60; w=120; sp=R['SPX']; spm=sp.rolling(w,min_periods=60).mean(); spvar=((sp-spm)**2).rolling(w,min_periods=60).mean(); B=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U:
 x=R[s]; xm=x.rolling(w,min_periods=60).mean(); B[s]=(((x-xm)*(sp-spm)).rolling(w,min_periods=60).mean()/spvar)
raw=R.rolling(m,min_periods=45).sum().sub(B.mul(sp.rolling(m,min_periods=45).sum(),axis=0))
for h in [1,5,10]:
 y=R.shift(-1).rolling(h).sum(); ic=[];ds=[];ns=[]
 for dt in R.index:
  z=pd.concat([raw.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 q=pd.Series(ic,index=ds); print('h',h,'dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  z=q[a:b]; print(a+'-'+b,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('coverage',raw.notna().mean().mean(),'rank_turnover',raw.rank(pct=True).diff().abs().mean().mean())
