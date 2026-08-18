import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}
c=pd.concat({s:x.close for s,x in d.items()},axis=1); v=pd.concat({s:x.volume for s,x in d.items()},axis=1)
r=np.log(c).diff(); ret5=c/c.shift(5)-1
# Volume-confirmed short-term reversal: fade recent moves, with unusually high volume
# as a confidence multiplier; signal is lagged one completed day.
vr=v/(v.rolling(20,min_periods=10).median())
f=(-ret5*vr.clip(0.5,3.0)).shift(1)
res=[]
for dt in r.index:
 x=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: res.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-06-10')]:
 z=x.loc[a:b]; print('regime',a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for dt in r.index:
  z=f.loc[dt]; yy=y.loc[dt]; ok=z.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(z[ok],yy[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
print('avg_valid_assets',x.n.mean(),'turnover_proxy',np.mean(np.sign(f).diff().abs().stack()>0))
f.to_csv('scripts/miner_3_20330610_volume_confirmed_reversal_signal.csv')
