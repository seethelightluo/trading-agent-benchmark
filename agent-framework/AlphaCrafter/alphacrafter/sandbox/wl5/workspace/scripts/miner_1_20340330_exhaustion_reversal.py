import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is not None and len(x)>150:
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index()
rows=[]
for s,x in D.items():
 c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float)
 loc=((c-l)/(h-l).replace(0,np.nan)*2-1)
 f=-(c.pct_change(5)/(c.pct_change().rolling(20).std()*np.sqrt(20)))*loc
 fr=c.shift(-10)/c-1
 rows.append(pd.DataFrame({'f':f,'fr':fr,'s':s}))
a=pd.concat(rows); ics=[]
for dt,g in a.groupby(level=0):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: ics.append((dt,g.f.corr(g.fr)))
ic=pd.Series(dict(ics)); ic=ic.dropna(); wide=a.reset_index().pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('dates',len(ic),'meanN',a.groupby(level=0).size().mean(),'coverage',len(a)/(len(set(a.index))*len(D)),'IC',ic.mean(),'ICIR_ann',ic.mean()/ic.std()*np.sqrt(252),'hit',(ic>0).mean(),'turnover',wide.diff().abs().mean().mean())
for lo,hi in [('2026-07-16','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2034-03-30')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi)]; print(lo,hi,len(q),q.mean())
print('decay')
for k in [5,10,20]:
 vals=[]
 for s,x in D.items():
  c=x.close.astype(float); h=x.high.astype(float); l=x.low.astype(float); loc=((c-l)/(h-l).replace(0,np.nan)*2-1)
  f=-(c.pct_change(5)/(c.pct_change().rolling(20).std()*np.sqrt(20)))*loc; vals.append(pd.DataFrame({'f':f,'fr':c.shift(-k)/c-1,'s':s}))
 z=pd.concat(vals).dropna(); q=[]
 for dt,g in z.groupby(level=0):
  if len(g)>=8:q.append(g.f.corr(g.fr))
 print(k,np.nanmean(q),len(q))
a.reset_index().rename(columns={'f':'signal','date':'asof_date'})[['asof_date','s','signal']].to_csv('scripts/miner_1_20340330_exhaustion_reversal_signal.csv',index=False)
