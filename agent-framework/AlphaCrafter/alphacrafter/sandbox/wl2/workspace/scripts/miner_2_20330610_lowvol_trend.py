import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1); r=np.log(c).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date); vix=vix.set_index('date').close.reindex(c.index).ffill()
# Low-volatility regime trend: lagged 60d return, risk scaled, active below lagged 60d VIX median.
ret=c/c.shift(60)-1; vol=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=(ret/vol).shift(1)
active=(vix.shift(1)<vix.shift(1).rolling(60,min_periods=40).median())
f=f.where(active)
res=[]
for dt in r.index:
 q=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=q.notna()&y.notna()
 if ok.sum()>=8:res.append((dt,spearmanr(q[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',15,'active_coverage',len(x)/len(r),'avg_n',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-06-10')]:
 z=x.loc[a:b];print('regime',a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 y=r.rolling(h).sum().shift(-h);qv=[]
 for dt in r.index:
  q=f.loc[dt]; yy=y.loc[dt];ok=q.notna()&yy.notna()
  if ok.sum()>=8:qv.append(spearmanr(q[ok],yy[ok]).statistic)
 print('decay',h,len(qv),np.mean(qv),np.mean(qv)/np.std(qv,ddof=1))
print('signal','scripts/miner_2_20330610_lowvol_trend_signal.csv');f.to_csv('scripts/miner_2_20330610_lowvol_trend_signal.csv')
