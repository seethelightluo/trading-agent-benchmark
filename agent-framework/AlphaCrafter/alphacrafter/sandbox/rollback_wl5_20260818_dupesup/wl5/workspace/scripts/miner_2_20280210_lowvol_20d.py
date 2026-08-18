import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-02-10')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x.date); x=x[x.date<=end].sort_values('date').set_index('date'); D[s]=x.close.astype(float)
pd0=pd.DataFrame(D).sort_index(); r=pd0.pct_change(); fac=-r.rolling(20).std(); fwd=pd0.shift(-1)/pd0-1
for h in [1,5,10]:
 q=pd0.shift(-h)/pd0-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print(h,{'dates':len(a),'mean_n':round(float(np.mean(ns)),2),'coverage':round(float(np.mean(ns)/15),4),'ic':round(float(np.mean(a)),6),'icir':round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit':round(float(np.mean(a>0)),4)})
print('period',fac.index.min().date(),fac.index.max().date())
