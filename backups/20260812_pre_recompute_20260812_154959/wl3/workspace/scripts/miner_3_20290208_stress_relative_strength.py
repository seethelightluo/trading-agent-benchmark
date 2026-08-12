import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date'); c=d.close
 frames.append(pd.DataFrame({'date':d.date,'symbol':s,'r20':c.pct_change(20),'fwd':c.shift(-1)/c-1}))
a=pd.concat(frames,ignore_index=True); w=a.pivot(index='date',columns='symbol',values='r20')
# Global equity stress: lagged average of SPX, NDX and 000300 20d returns. Defensive relative strength is most valuable in stress.
eq=[x for x in ['SPX','NDX','000300.SH'] if x in w.columns]; stress=(w[eq].mean(axis=1)<0).shift(1)
# conditional signal uses lagged r20 already, cross-sectional median normalization
x=w.sub(w.median(axis=1),axis=0)
a['factor_value']=a.apply(lambda z: x.loc[z.date,z.symbol] if bool(stress.get(z.date,False)) else 0.5*x.loc[z.date,z.symbol],axis=1)
a=a.dropna(subset=['factor_value','fwd']);a.to_csv('scripts/miner_3_20290208_stress_relative_strength_signal.csv',index=False)
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.fwd.nunique()>1: ics.append((pd.Timestamp(dt),g.factor_value.corr(g.fwd,method='spearman')))
ic=pd.Series(dict(ics)).sort_index();print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2));print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4));p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True);print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4),'cutoff',ic.index.max().date())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent252',None,None)]:
 q=ic if label!='recent252' else ic.tail(252)
 if label!='recent252':q=q.loc[lo:hi]
 print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
