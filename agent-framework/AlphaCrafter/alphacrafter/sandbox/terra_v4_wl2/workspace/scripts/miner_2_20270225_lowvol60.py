import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); d[s]=x.close.astype(float)
p=pd.concat(d,axis=1).sort_index(); r=p.pct_change(); f=-r.rolling(60,min_periods=40).std()
for h in [1,5,10]:
 y=p.shift(-h)/p-1; a=[]; n=[]
 for i in range(60,len(p)-h):
  z=pd.concat([f.iloc[i],y.iloc[i+h-1]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 a=np.array(a);print(h,len(a),np.mean(n),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
print('coverage',f.notna().mean().mean())
