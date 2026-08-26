import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=180: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Defensive quality: medium-term return rewarded, but penalize downside volatility; all windows causal.
mom=P.pct_change(40)
down=R.where(R<0,0).pow(2).rolling(60,min_periods=40).mean().pow(.5)
tot=R.pow(2).rolling(60,min_periods=40).mean().pow(.5)
# downside-risk-adjusted momentum, mildly reward low downside/total-vol quality
f=(mom/(down+1e-8))*(1+0.25*(1-down/(tot+1e-8)).clip(-2,2))
fr=P.pct_change(10).shift(-10)
ics=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);dates.append(dt);ns.append(len(z));ranks.append(f.loc[dt].rank(pct=True))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
a=np.array(ics); S=pd.DataFrame(ranks,index=dates)
pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_2_20330818_downside_quality_momentum_signal.csv',index=False)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(S.diff().abs().mean().mean()),6)})
for x,y in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-08-17')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))];print(x,len(z),round(float(z.mean()),6),round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6))
