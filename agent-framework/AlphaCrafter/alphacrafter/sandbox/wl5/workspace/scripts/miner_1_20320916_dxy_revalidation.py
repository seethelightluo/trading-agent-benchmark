import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];xs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): d=d.copy();d.date=pd.to_datetime(d.date);xs[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill();r=p.pct_change()
mc=pd.read_csv('../persistent/index_data/DXY.csv');mc.date=pd.to_datetime(mc.date);dxy=mc.set_index('date').close.astype(float).sort_index().reindex(p.index).ffill().pct_change()
rows=[];art=[]
for i in range(100,len(p)-10):
 dt=p.index[i]; rr=r.iloc[:i+1]; w=dxy.iloc[:i+1].iloc[-60:]; rw=rr.iloc[-60:].loc[w.index]; var=w.var()
 beta=rw.apply(lambda x:x.cov(w)/var if var>1e-12 else np.nan)
 vol=rr.iloc[-20:].std()*np.sqrt(252); rev=-p.iloc[i]/p.iloc[i-10]-1; sig=rev/(vol.replace(0,np.nan))
 ab=beta.abs(); fac=sig*(1-(ab-ab.median())/(ab.std()+1e-9)*.25)
 y=p.iloc[i+10]/p.iloc[i]-1;q=pd.DataFrame({'f':fac,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  ic=q.f.corr(q.y);rows.append((dt,float(ic),len(q)))
  for s,v in fac.items():
   if s in q.index:art.append({'date':dt,'symbol':s,'signal':float(v),'ic':float(ic)})
a=np.array([x[1] for x in rows]);print('dates',len(a),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15);print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for st,en in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-09-15')]:
 v=np.array([x[1] for x in rows if pd.Timestamp(st)<=x[0]<=pd.Timestamp(en)]);print(st,len(v),v.mean(),v.mean()/v.std(ddof=1))
pd.DataFrame(art).to_csv('scripts/miner_1_20320916_dxy_revalidation_signal.csv',index=False)
