import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-11-28']
r=P.pct_change(); hi=P.rolling(120,min_periods=80).max().shift(1); vol=r.rolling(20,min_periods=15).std().shift(1); f=-(P.shift(1)/hi-1)/vol.replace(0,np.nan)
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(rows).dropna(); print(h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for nm,q in [('r360',slice('2028-11-28','2029-11-28')),('r180',slice('2029-05-28','2029-11-28'))]:
 fw=P.shift(-10)/P-1; rows=[]
 for d in P.index:
  if not (q.start<=str(d.date())<=q.stop): continue
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(rows); print(nm,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
