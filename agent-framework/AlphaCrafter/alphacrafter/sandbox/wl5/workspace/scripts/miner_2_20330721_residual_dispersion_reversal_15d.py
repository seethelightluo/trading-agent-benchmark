import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=260: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
disp=R.std(axis=1); gate=disp>disp.rolling(126,min_periods=80).quantile(.70)
for s in P.columns:
 beta=R[s].rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var()
 resid=R[s]-beta*m
 shock=resid.rolling(8,min_periods=8).sum()
 rv=resid.rolling(45,min_periods=30).std()
 # Reversal is strongest after a normalized idiosyncratic shock, amplified in broad dispersion.
 f[s]=-(shock/(rv*np.sqrt(8)+1e-12))*(1+0.35*gate.astype(float))
fr=R.shift(-15).rolling(15,min_periods=15).sum(); ics=[]; dates=[]; ns=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-07-20')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6) if len(z)>1 else None)
out='scripts/miner_2_20330721_residual_dispersion_reversal_15d_signal.csv'; f.loc[dates].to_csv(out,index_label='date'); print('signal_artifact',out)
