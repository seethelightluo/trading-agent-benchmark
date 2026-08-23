import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
px=pd.concat(D,axis=1).sort_index(); ret=px.pct_change(); vol=ret.rolling(60).std(); f=(px/px.shift(20)-1)/vol
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(px.shift(-h).loc[d]/px.loc[d]-1)],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):I.append(x);N.append(len(z));ds.append(d)
 a=np.array(I);print(h,len(a),np.mean(N),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
 if h==10:
  for y in sorted(set(d.year for d in ds)):
   b=a[[d.year==y for d in ds]];print(y,round(b.mean(),5),len(b))
