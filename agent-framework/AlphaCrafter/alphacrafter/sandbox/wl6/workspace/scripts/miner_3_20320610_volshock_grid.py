import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:'2032-06-09']; r=P.pct_change()
for a,b in [(12,60),(15,80),(15,100),(18,80),(18,100),(20,60),(20,100),(25,100)]:
 f=-(r.rolling(a,min_periods=max(8,a//2)).std()/(r.rolling(b,min_periods=max(20,b//2)).std()+1e-12)-1); fr=P.shift(-20)/P-1; x=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=np.array(x); print(a,b,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
