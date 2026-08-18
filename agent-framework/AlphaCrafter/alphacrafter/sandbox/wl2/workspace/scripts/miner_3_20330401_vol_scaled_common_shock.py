import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p):return None
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close.sort_index()
p=pd.concat({s:ld(s) for s in U},axis=1); r=np.log(p).diff(); vol=r.rolling(20,min_periods=15).std();
# cross-sectional residual shock: remove daily common median, normalize by own vol
shock=(r.rolling(5).sum()-r.rolling(5).sum().median(axis=1).values[:,None]).div(vol*np.sqrt(5))
f=-shock.shift(1)
res=[]
for d in r.index:
 v=f.loc[d];y=r.shift(-1).loc[d];ok=v.notna()&y.notna()
 if ok.sum()>=8:res.append((d,spearmanr(v[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2026','2029-12-31'),('2030','2033-04-01')]:
 z=x.loc[a:b];print(a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1))
print('decay')
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h);q=[]
 for d in r.index:
  v=f.loc[d];yy=y.loc[d];ok=v.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],yy[ok]).statistic)
 print(h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_3_20330401_vol_scaled_common_shock_signal.csv')
