import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; hi={}; lo={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=100:
  d=d.set_index('date'); cl[s]=d.close.astype(float); hi[s]=d.high.astype(float); lo[s]=d.low.astype(float)
P=pd.DataFrame(cl).sort_index(); H=pd.DataFrame(hi).reindex(P.index); L=pd.DataFrame(lo).reindex(P.index)
prev=P.shift(1); tr=(pd.concat([H-L,(H-prev).abs(),(L-prev).abs()],axis=1).to_numpy().reshape(len(P),3,-1).max(axis=1)); tr=pd.DataFrame(tr,index=P.index,columns=P.columns)
atr=(tr/P.shift(1)).rolling(20,min_periods=15).mean(); f=-P.pct_change(30)/atr.replace(0,np.nan); fr=P.pct_change(10).shift(-10)
ics=[]; dates=[]; ns=[]; ranks=[]; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(z)); ranks.append(f.loc[dt].rank(pct=True))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append((dt,s,float(f.loc[dt,s])))
a=np.array(ics); out='scripts/miner_2_20330929_range_normalized_reversal_signal.csv'; pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv(out,index=False)
print({'dates':len(a),'start':str(dates[0].date()),'end':str(dates[-1].date()),'mean_n':round(float(np.mean(ns)),3),'coverage':round(float(np.mean(ns)/15),6),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),6),'hit':round(float(np.mean(a>0)),6),'turnover':round(float(pd.DataFrame(ranks,index=dates).diff().abs().mean().mean()),6),'signal_artifact':out})
for x,y in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-09-28')]:
 z=a[(np.array(dates)>=pd.Timestamp(x))&(np.array(dates)<=pd.Timestamp(y))]
 if len(z)>1: print(x,len(z),round(float(z.mean()),6),round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),6))
