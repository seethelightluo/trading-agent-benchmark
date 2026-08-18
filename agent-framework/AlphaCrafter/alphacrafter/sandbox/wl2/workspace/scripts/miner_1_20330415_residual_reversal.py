import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date').close.sort_index()
p=pd.concat({s:load(s) for s in U},axis=1);p=p.loc[:'2033-04-15'];r=np.log(p).diff()
# lagged cross-sectional residual reversal of 5d return, volatility normalized
m=r.rolling(5,min_periods=5).sum(); med=m.median(axis=1); v=r.rolling(20,min_periods=15).std()*np.sqrt(5)
f=-(m.sub(med,axis=0)).div(v).shift(1)
rows=[]
for d in r.index:
 x=f.loc[d];y=r.shift(-1).loc[d];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'avg_n',x.n.mean(),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-04-15')]:
 z=x.loc[a:b];print('regime',a,b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1))
for h in [3,5,10]:
 y=r.rolling(h).sum().shift(-h);q=[]
 for d in r.index:
  a=f.loc[d];b=y.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_1_20330415_residual_reversal_signal.csv')
