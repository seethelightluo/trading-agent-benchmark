import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date');c=d.close.astype(float);r=np.log(c/c.shift(1))
 eff=(r.rolling(20,min_periods=18).sum()/(r.abs().rolling(20,min_periods=18).sum()+1e-12)).shift(1)
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'factor_value':eff,'r1':c.shift(-1)/c-1,'r3':c.shift(-3)/c-1,'r5':c.shift(-5)/c-1,'r10':c.shift(-10)/c-1}))
a=pd.concat(rows,ignore_index=True)
for h in [1,3,5,10]:
 q=[]
 for dt,g in a.dropna(subset=['factor_value','r%d'%h]).groupby('date'):
  if len(g)>=8 and g.factor_value.nunique()>1:q.append((pd.Timestamp(dt),g.factor_value.corr(g['r%d'%h],method='spearman'),len(g)))
 z=pd.DataFrame(q,columns=['date','ic','n']).set_index('date');ic=z.ic
 print('H',h,'dates',len(ic),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'cov',round(z.n.mean()/15,4))
 for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028-29','2028','2029-12-31'),('recent120',None,None)]:
  v=ic.tail(120) if lab=='recent120' else ic.loc[lo:hi];print(' ',lab,len(v),round(v.mean(),6),round(v.mean()/v.std(ddof=1),6) if len(v)>1 else None)
rk=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('artifact_rows',len(a),'dates',a.date.nunique(),'instruments',a.symbol.nunique(),'coverage',round(a.dropna(subset=['factor_value']).shape[0]/(a.date.nunique()*15),4),'turnover_proxy',round(rk.diff().abs().mean(axis=1).mean(),4),'cutoff',a.date.max().date());a.to_csv('scripts/miner_2_20290614_path_efficiency_signal.csv',index=False)
