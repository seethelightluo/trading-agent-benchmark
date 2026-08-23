import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index(); r=np.log(P/P.shift(1))
# volatility acceleration: recent realized volatility relative to medium-term volatility, lagged one session
f=(r.rolling(20).std()/r.rolling(60).std()).shift(1); fw=P.shift(-20)/P-1
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals);a=a[np.isfinite(a)];print({'h':h,'dates':len(a),'avg_n':round(np.mean(ns),3),'coverage':round(np.mean(ns)/15,4),'IC':round(np.mean(a),6),'ICIR':round(np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),3),'hit':round(np.mean(a>0),4)})
# 20d regime diagnostics
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for lo,hi in [('2020-01-01','2027-12-31'),('2028-01-01','2031-12-31'),('2032-01-01','2035-01-03')]:
 q=[v for d,v in rows if lo<=str(d.date())<=hi];print('regime',lo,hi,'dates',len(q),'IC',round(float(np.mean(q)),6))
out=[]
for dt in f.index:
 for s in f.columns:
  if pd.notna(f.loc[dt,s]):out.append({'date':str(dt.date()),'symbol':s,'signal':float(f.loc[dt,s])})
pd.DataFrame(out).to_csv('scripts/miner_1_20350104_volatility_acceleration_signal.csv',index=False)
