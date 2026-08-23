import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2032-06-09'
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:cut]; r=P.pct_change()
for a,b in [(10,60),(15,60),(20,80),(10,40),(20,120)]:
 v1=r.rolling(a,min_periods=max(8,a//2)).std(); v2=r.rolling(b,min_periods=max(20,b//2)).std(); f=-(v1/(v2+1e-12)-1)
 for h in [10,20]:
  fr=P.shift(-h)/P-1; zlist=[]
  for dt in P.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: zlist.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  x=np.array(zlist); print(a,b,h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
