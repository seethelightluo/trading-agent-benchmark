import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1); r=np.log(c).diff()
vol=r.rolling(20,min_periods=12).std(); daily_disp=r.std(axis=1); disp=daily_disp.rolling(20,min_periods=12).mean()
th=disp.rolling(60,min_periods=30).median(); active=(disp>th).astype(float)
f=(-r.rolling(3,min_periods=3).sum()/vol).clip(-5,5).mul(active,axis=0).shift(1).replace([np.inf,-np.inf],np.nan)
res=[]
for dt in r.index:
 x=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: res.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',15,'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'avg_n',x.n.mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-08-18')]:
 z=x.loc[a:b]; print('regime',a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for dt in r.index:
  a=f.loc[dt]; yy=y.loc[dt]; ok=a.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],yy[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
print('turnover_proxy',np.mean(np.sign(f).diff().abs().stack()>0))
f.to_csv('scripts/miner_1_20330819_dispersion_conditioned_reversal_signal.csv')
