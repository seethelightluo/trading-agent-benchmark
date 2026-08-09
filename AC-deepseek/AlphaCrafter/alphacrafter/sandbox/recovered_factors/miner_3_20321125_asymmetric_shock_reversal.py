import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in watch:
 f='../persistent/stock_data/'+s+'.csv'
 x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); x=x.set_index('date').sort_index()
 d[s]=x.close.astype(float)
px=pd.DataFrame(d).sort_index(); ret=px.pct_change()
# asymmetric short shock reversal: reverse recent 3d return, scaled by trailing downside deviation
# all inputs shifted one day at signal formation
down=ret.where(ret<0).rolling(20,min_periods=10).std()
sig=-(ret.rolling(3,min_periods=3).sum()/ (down.rolling(3,min_periods=3).mean()+1e-8)).shift(1)
# winsorize cross section each date
sig=sig.clip(-10,10)
print('rows',len(px),'assets',len(watch),'signal cells',int(sig.notna().sum().sum()),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h)
 vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
# turnover rank 10d
r=sig.rank(axis=1,pct=True); print('turn10',np.nanmean(np.abs(r-r.shift(10)).mean(axis=1)))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 z=[]; fwd=px.pct_change(5).shift(-5)
 for dt in sig.loc[lo:hi].index:
  a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 print(lo,hi,'n',len(z),'IC',round(np.mean(z),6) if z else None,'ICIR',round(np.mean(z)/np.std(z,ddof=1),6) if len(z)>1 else None)
