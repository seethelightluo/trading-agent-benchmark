import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change()
 # Candidate: short reversal conditioned on unusually wide recent range.
 # Range shock makes the reversal signal stronger, while volatility scaling
 # normalizes cross-asset risk.
 vol=r.rolling(20,min_periods=10).std()
 shock=(d.high.astype(float)-d.low.astype(float))/c
 shock_score=shock.rolling(5,min_periods=3).mean()/shock.rolling(60,min_periods=20).mean()
 x=(-c.pct_change(3)/vol)*shock_score
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'factor_value':x,'fwd':c.shift(-1)/c-1}))
a=pd.concat(rows,ignore_index=True).dropna(subset=['factor_value','fwd'])
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor_value.nunique()>1 and g.fwd.nunique()>1:
  ics.append((pd.Timestamp(dt),g.factor_value.corr(g.fwd,method='spearman')))
ic=pd.Series(dict(ics)).sort_index()
a.to_csv('scripts/miner_3_20290503_range_shock_reversal3_signal.csv',index=False)
print('artifact_rows',len(a),'dates',len(ic),'instruments',a.symbol.nunique(),'avg_n',round(a.groupby('date').size().mean(),2))
print('IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4))
p=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True)
print('turnover_proxy',round((p.diff().abs()>0.2).mean(axis=1).mean(),4),'coverage',round(len(a)/(len(ic)*15),4),'cutoff',ic.index.max().date())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent252',None,None)]:
 q=ic.tail(252) if label=='recent252' else ic.loc[lo:hi]
 print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
for h in [1,3,5]:
 z=[]
 for s in U:
  d=get_stock_daily_data(s,4000)
  if d is None or len(d)==0:d=get_index_daily_data(s,4000)
  if d is None or len(d)==0:continue
  d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=10).std(); sh=((d.high.astype(float)-d.low.astype(float))/c); ss=sh.rolling(5,min_periods=3).mean()/sh.rolling(60,min_periods=20).mean(); f=(-c.pct_change(3)/vol)*ss
  z.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'x':f,'y':c.shift(-h)/c-1}))
 z=pd.concat(z); ii=[]
 for dt,g in z.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.x.nunique()>1 and g.y.nunique()>1: ii.append(g.x.corr(g.y,method='spearman'))
 print('decay',h,'IC',round(np.nanmean(ii),6),'n',len(ii),'ICIR',round(np.nanmean(ii)/np.nanstd(ii),6))
