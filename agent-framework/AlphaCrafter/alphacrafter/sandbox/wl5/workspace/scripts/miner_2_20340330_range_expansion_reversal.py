import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=160: D[s]=d.set_index('date')[['open','close','high','low']].astype(float)
P=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); R=P.pct_change()
# Range-expansion reversal: oversold 10d move, amplified when recent true range expands versus its medium baseline.
F={}
for s,d in D.items():
 rng=(d.high-d.low).abs(); atr20=rng.rolling(20,min_periods=15).mean(); atr80=rng.rolling(80,min_periods=60).mean()
 expansion=(atr20/(atr80+1e-9)).clip(.5,2.5)
 ret10=d.close/d.close.shift(10)-1
 F[s]=(-(ret10/(R[s].rolling(40,min_periods=30).std()*np.sqrt(10)+1e-9))*expansion.pow(.5)).clip(-6,6)
F=pd.DataFrame(F).sort_index(); rows=[]
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; a=[]; dates=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);dates.append(dt);ns.append(len(z))
 a=np.array(a); dates=pd.DatetimeIndex(dates)
 print('horizon',h,'dates',len(a),'start',str(dates[0].date()),'end',str(dates[-1].date()),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-03-01')]:
  z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
 if h==10:
  for dt in dates:
   for s in F.columns:
    if pd.notna(F.loc[dt,s]): rows.append((dt,s,float(F.loc[dt,s])))
  S=pd.DataFrame([F.loc[d].rank(pct=True) for d in dates],index=dates); print('turnover',round(S.diff().abs().mean().mean(),6))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20340330_range_expansion_reversal_signal.csv',index=False)
