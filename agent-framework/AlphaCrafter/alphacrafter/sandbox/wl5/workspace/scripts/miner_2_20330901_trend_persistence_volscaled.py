import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Defensive trend persistence: reward sustained positive daily breadth over 60d,
# scaled by recent volatility; all inputs are known at signal date.
trend=P.pct_change(60); persistence=(R>0).rolling(60,min_periods=45).mean()-0.5
vol=R.rolling(30,min_periods=20).std()
f=(trend*(1+0.8*persistence))/vol.replace(0,np.nan)
fr=P.pct_change(10).shift(-10)
ics=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z)); ranks.append(f.loc[dt].rank(pct=True))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
out='scripts/miner_2_20330901_trend_persistence_volscaled_signal.csv'
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv(out,index=False)
def stats(z):
 return (len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float(np.mean(z>0)))
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6),'signal_artifact':out})
for x,y in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-08-31')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]; print(x,stats(z))
for h in [5,10,20]:
 q=P.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'dates',len(vals),'IC',round(float(np.nanmean(vals)),6))
