import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2029-10-03')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U},axis=1).sort_index().ffill().loc[:E]; r=P.pct_change(); disp=r.rolling(20).std().mean(1); med=disp.rolling(252,min_periods=126).median()
for look in [3,5,10]:
 for vol in [10,20,40]:
  f=-r.rolling(look).sum()/r.rolling(vol).std(); f=f.mul(disp/med,axis=0)
  for h in [5,10]:
   a=[]
   for i in range(len(P)-h):
    z=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=z.notna()&y.notna()
    if ok.sum()>=8:
     q=pd.Series(z[ok]).corr(pd.Series(y[ok]),method='spearman')
     if np.isfinite(q):a.append(q)
   a=np.array(a); print(look,vol,h,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),4),round((a>0).mean(),3))
