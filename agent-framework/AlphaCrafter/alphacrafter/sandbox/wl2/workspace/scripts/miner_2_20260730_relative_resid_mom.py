import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
R=pd.DataFrame({s:D[s].close.pct_change() for s in U}).sort_index()
# candidate: lagged cross-sectional residual of medium-short return, with market median removed
for look in [3,7,12,20]:
 F=R.rolling(look,min_periods=max(2,look//2)).sum().sub(R.rolling(look,min_periods=max(2,look//2)).sum().median(axis=1),axis=0).shift(1)
 print('LOOK',look)
 for h in [1,5,10]:
  Y=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[];ns=[]
  for dt in F.index:
   z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
   if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
  q=np.array(q); print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
 print('coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
