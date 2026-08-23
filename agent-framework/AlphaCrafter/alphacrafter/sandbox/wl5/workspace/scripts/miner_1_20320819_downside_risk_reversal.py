import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);xs[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill();r=p.pct_change(); rows=[]; art=[]
for i in range(100,len(p)-10):
 dt=p.index[i]; rr=r.iloc[:i+1]
 ret=rr.iloc[-20:]; down=ret.clip(upper=0).std().replace(0,np.nan)*np.sqrt(252)
 total=ret.std().replace(0,np.nan)*np.sqrt(252)
 # Favor oversold assets with smaller downside risk; interpretable downside-risk-adjusted reversal.
 sig=-(p.iloc[i]/p.iloc[i-10]-1)/(down+0.5*total)
 y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.DataFrame({'f':sig,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  ic=z.f.corr(z.y);rows.append((dt,float(ic),len(z)))
  for s,v in sig.items():
   if s in z.index:art.append({'date':dt,'symbol':s,'signal':float(v),'ic':float(ic)})
q=np.array([x[1] for x in rows]);print('dates',len(q),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15);print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-08-18')]:
 v=np.array([x[1] for x in rows if pd.Timestamp(a)<=x[0]<=pd.Timestamp(b)]);print(a,len(v),v.mean() if len(v) else np.nan,v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
pd.DataFrame(art).to_csv('scripts/miner_1_20320819_downside_risk_reversal_signal.csv',index=False)
