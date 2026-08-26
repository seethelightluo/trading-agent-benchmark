import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=140: D[s]=d.set_index('date')[['open','close','high','low']].astype(float)
# Candle pressure: signed close location within daily range, weighted by range relative to 20d ATR; reverse persistent buying/selling pressure.
vals={}
for s,d in D.items():
 rng=(d.high-d.low).replace(0,np.nan)
 clv=((d.close-d.open)/rng).clip(-1,1)
 atr=rng.rolling(20,min_periods=15).mean()
 vals[s]=-(clv*(rng/(atr+1e-9)).clip(0,3)).rolling(5,min_periods=4).mean()
F=pd.DataFrame(vals).clip(-3,3).sort_index(); P=pd.DataFrame({s:d.close for s,d in D.items()}).reindex(F.index)
rows=[]
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; ics=[]; dates=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z))
 a=np.array(ics); dates=pd.DatetimeIndex(dates)
 print('horizon',h,'dates',len(a),'start',str(dates[0].date()),'end',str(dates[-1].date()),'mean_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(a>0),6))
 for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-02-28')]:
  z=a[(dates>=pd.Timestamp(x))&(dates<=pd.Timestamp(y))]
  if len(z)>1: print('regime',x,len(z),round(z.mean(),6))
 if h==10:
  for dt in dates:
   for s in F.columns:
    if pd.notna(F.loc[dt,s]): rows.append((dt,s,float(F.loc[dt,s])))
S=pd.DataFrame([F.loc[d].rank(pct=True) for d in dates],index=dates); print('turnover',round(S.diff().abs().mean().mean(),6))
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20340302_candle_pressure_reversal_signal.csv',index=False)
