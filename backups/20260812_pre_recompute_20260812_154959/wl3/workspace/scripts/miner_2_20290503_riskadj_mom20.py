import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change()
 # Risk-adjusted medium-term momentum, using only completed sessions.
 mom=c.pct_change(20).shift(1); vol=r.rolling(20,min_periods=18).std().shift(1)
 sig=mom/(vol*np.sqrt(20)+1e-12)
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'factor_value':sig,'forward_return_1d':c.shift(-1)/c-1,'forward_return_3d':c.shift(-3)/c-1,'forward_return_5d':c.shift(-5)/c-1,'forward_return_10d':c.shift(-10)/c-1}))
a=pd.concat(rows).dropna(subset=['factor_value','forward_return_1d'])
out='scripts/miner_2_20290503_riskadj_mom20_signal.csv'; a.to_csv(out,index=False)
for h in [1,3,5,10]:
 col='forward_return_%dd'%h; q=[]
 for dt,g in a.dropna(subset=[col]).groupby('date'):
  if len(g)>=8 and g.factor_value.nunique()>1: q.append((pd.Timestamp(dt),g.factor_value.corr(g[col],method='spearman')))
 ic=pd.Series(dict(q)).sort_index(); print('H',h,'dates',len(ic),'avg_n',round(a.groupby('date').size().mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(),6),'hit',round((ic>0).mean(),4))
 for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31'),('recent120',None,None)]:
  v=ic.tail(120) if label=='recent120' else ic.loc[lo:hi]; print(' ',label,len(v),round(v.mean(),6),round(v.mean()/v.std(),6) if len(v)>1 else None)
rk=a.pivot(index='date',columns='symbol',values='factor_value').rank(axis=1,pct=True)
print('artifact_rows',len(a),'dates',a.date.nunique(),'instruments',a.symbol.nunique(),'coverage',round(len(a)/(a.date.nunique()*15),4),'turnover_proxy',round((rk.diff().abs()>0.2).mean(axis=1).mean(),4),'cutoff',a.date.max().date())
