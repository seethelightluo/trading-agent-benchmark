import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1); r=np.log(c).diff()
# Relative-strength spread: asset's trailing 20d return minus the equal-weight universe trailing return,
# scaled by its trailing 20d daily volatility; lag one completed day.
ret20=c/c.shift(20)-1; bench=ret20.mean(axis=1); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=((ret20.sub(bench,axis=0))/vol).shift(1)
res=[]
for dt in r.index:
 v=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=v.notna()&y.notna()
 if ok.sum()>=8: res.append((dt,spearmanr(v[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-05-27')]:
 z=x.loc[a:b]; print('regime',a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for dt in r.index:
  v=f.loc[dt]; yy=y.loc[dt]; ok=v.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],yy[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
print('avg_valid_assets',x.n.mean())
f.to_csv('scripts/miner_3_20330527_relative_strength_spread_signal.csv')
