import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=160: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r10=P/P.shift(10)-1; v20=R.rolling(20,min_periods=15).std(); v120=R.rolling(120,min_periods=80).std()
compression=(v120/(v20+1e-8)).clip(0.5,2.5)
# Compression-conditioned reversal: reverse recent trend, with stronger reversal after volatility compression.
f=(-(r10/(v20*np.sqrt(20)+1e-8)))*compression.pow(0.5); f=f.clip(-6,6)
fwds={h:P.shift(-h)/P-1 for h in [5,10,20]}; rows=[]; out={}
for h in [5,10,20]:
 a=[]; ds=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fwds[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c); ds.append(dt); ns.append(len(z))
 out[h]=(np.array(a),pd.DatetimeIndex(ds),ns)
 for s in f.columns:
  if h==10 and pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340216_compression_reversal_signal.csv',index=False)
for h,(a,ds,ns) in out.items():
 print('horizon',h,'dates',len(a),'start',str(ds[0].date()),'end',str(ds[-1].date()),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2026-08-31','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-02-15')]:
  z=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
S=pd.DataFrame([f.loc[d].rank(pct=True) for d in out[10][1]],index=out[10][1]); print('turnover',round(S.diff().abs().mean().mean(),6))
