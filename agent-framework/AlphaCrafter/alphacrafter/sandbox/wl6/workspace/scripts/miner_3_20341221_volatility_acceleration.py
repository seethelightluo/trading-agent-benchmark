import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
    d=get_stock_daily_data(s, days=5000)
    if d is None or len(d)<100: d=get_index_daily_data(s, days=5000)
    if d is None: return None
    d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
P=pd.concat({s:load(s) for s in U if load(s) is not None},axis=1).sort_index()
r=np.log(P/P.shift(1)); v=r.rolling(20).std()*np.sqrt(20); vlong=r.rolling(60).std()*np.sqrt(60)
# lagged signal, cross-sectional median demeaned
f=(v/vlong).shift(1)
rows=[]
for h in [5,10,20,40]:
  fw=P.shift(-h)/P-1
  ics=[]; n=[]; cov=[]
  for dt in f.index:
    x=f.loc[dt]; y=fw.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z)); cov.append(len(z)/len(U))
  a=np.array(ics); a=a[np.isfinite(a)]
  print({'h':h,'dates':len(a),'avg_n':round(float(np.mean(n)),3),'coverage':round(float(np.mean(cov)),4),'IC':round(float(np.mean(a)),6),'ICIR':round(float(np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a))),3),'hit':round(float(np.mean(a>0)),4)})
# regimes at h20
fw=P.shift(-20)/P-1; a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
for lo,hi in [('2020','2027-12-31'),('2028','2031-12-31'),('2032','2034-12-06')]:
 q=[x[1] for x in a if str(x[0])>=lo and str(x[0])<=hi];print('regime',lo,hi,len(q),round(float(np.mean(q)),6) if q else None)
# write artifact
out=[]
for dt in f.index:
 for s in f.columns:
  if pd.notna(f.loc[dt,s]):out.append({'date':str(dt.date()),'symbol':s,'signal':float(f.loc[dt,s])})
pd.DataFrame(out).to_csv('scripts/miner_3_20341221_volatility_acceleration_signal.csv',index=False)
