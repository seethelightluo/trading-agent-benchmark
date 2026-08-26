import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-07-08')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float).sort_index() for s in U}
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); ics=[]; dates=[]; ns=[]; out=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  i=p.index.get_loc(dt)
  if i<125 or i+20>=len(p): continue
  r=p.pct_change(); ret60=p.iloc[i]/p.iloc[i-60]-1; peak=p.iloc[i-60:i+1].max(); dd=p.iloc[i]/peak-1; vol=r.iloc[i-40:i].std()*np.sqrt(40)
  # recovery-adjusted trend: positive long-term return penalized by distance below recent high
  vals[s]=(ret60 + 0.5*dd)/max(vol,1e-6); fw[s]=p.iloc[i+20]/p.iloc[i]-1
 if len(vals)<8: continue
 a=np.array(list(vals.values())); b=np.array([fw[s] for s in vals]); ic=spearmanr(a,b).statistic
 if np.isfinite(ic): ics.append(ic);dates.append(dt);ns.append(len(a))
 for s,v in vals.items():out.append({'date':dt.date().isoformat(),'symbol':s,'signal':v})
x=np.array(ics); ds=np.array(dates,dtype='datetime64[ns]'); print('factor=recovery-adjusted trend H20 cut',cut.date(),'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-07-08'),('2034-07-01','2035-07-08')]:
 z=x[(ds>=np.datetime64(a))&(ds<=np.datetime64(b))];print(a,b,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round(np.mean(z>0),3))
pd.DataFrame(out).to_csv('scripts/miner_1_20350709_recovery_trend_signal.csv',index=False)
