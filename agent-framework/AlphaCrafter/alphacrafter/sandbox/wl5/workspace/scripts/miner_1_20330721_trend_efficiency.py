import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Trend efficiency: signed return divided by path length; causal, bounded, volume-free.
ret= P.pct_change(20)
path=R.abs().rolling(20,min_periods=15).sum()
f=ret/path
fr=P.pct_change(10).shift(-10) # close-to-close forward 10d return
ics=[]; dates=[]; ns=[]; ranks=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z)); ranks.append(f.loc[dt].rank(pct=True))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-07-10')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 print(x,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6) if len(z)>1 else None)
out=f.loc[dates].copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20330721_trend_efficiency_signal.csv')
