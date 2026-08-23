import os,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close.sort_index()
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); market=r.mean(axis=1)
# Relative 20d return versus contemporaneous equal-weight benchmark, faded and volatility scaled.
rel=(p/p.shift(20)-1).sub((market.rolling(20).sum()),axis=0)
f=-rel/r.rolling(20).std()
y=p.shift(-10)/p-1
out=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=a.loc['2020-01-01':'2030-10-02']
mu=a.ic.mean(); sd=a.ic.std(ddof=1)
print(json.dumps({'factor':'relative_reversal','valid_dates':len(a),'average_instruments':a.n.mean(),'coverage':a.n.mean()/15,'ic':mu,'icir':mu/sd*np.sqrt(252),'hit':(a.ic>0).mean(),'regimes':{str(y):a.loc[a.index.year==y].ic.mean() for y in sorted(a.index.year.unique())}},default=str))
