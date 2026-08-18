import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); lr=np.log(p/p.shift(1)); r5=lr.rolling(5).sum(); v20=lr.rolling(20).std()
# high-dispersion short-term reversal: normalize reversal by own risk and activate above median cross-sectional dispersion
base=-(r5/v20); disp=lr.rolling(20).std().median(axis=1)
f=base.where(disp>disp.rolling(120).median(),0.0).rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
ics=[]; ns=[]; prev=None; tos=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],(p.shift(-10)/p-1).loc[dt]],axis=1).dropna()
 if len(a)>=8:
  ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a)); sig=f.loc[dt].rank(pct=True)
  if prev is not None:tos.append(np.nanmean(abs(sig-prev)))
  prev=sig
z=pd.Series(ics).dropna()
def st(x):return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC/ICIR/hit',st(z),'turnover',np.nanmean(tos))
for k in [120,252,504]:print('recent',k,st(z.tail(k)))
print('last',p.index[-1])
