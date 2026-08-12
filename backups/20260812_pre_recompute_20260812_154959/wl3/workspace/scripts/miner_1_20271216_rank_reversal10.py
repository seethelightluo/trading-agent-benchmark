import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is None: continue
 d=d.sort_values('date')
 c=d.close.astype(float)
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'symbol':s,
   'raw':-(c/c.shift(10)-1), 'fr':c.shift(-1)/c-1}))
x=pd.concat(rows,ignore_index=True)
x['f']=x.groupby('date')['raw'].rank(pct=True,method='average')
out=x[['date','symbol','f']].dropna()
out.to_csv('scripts/miner_1_20271216_rank_reversal10_signal.csv',index=False)
ics=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fr'])
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
  ics.append(g.f.corr(g.fr)); ns.append(len(g))
a=pd.Series(ics).dropna()
# signal turnover: rank changes relative to prior common date, averaged cross-section
p=out.pivot(index='date',columns='symbol',values='f').sort_index()
to=(p.diff().abs().mean(axis=1)/2).mean()
print('dates',len(a),'avg_n',round(float(np.mean(ns)),3),'min_n',min(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',np.mean(np.array(ns)/15),'turnover_proxy',to,'recent500',a.tail(500).mean())
for h in [3,5,10]:
 # same factor against h-day forward returns
 y=[]
 for s in U:
  d=get_stock_daily_data(s,4000)
  if d is None:d=get_index_daily_data(s,4000)
  if d is not None:
   c=d.sort_values('date').close.astype(float); y.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'symbol':s,'frh':c.shift(-h)/c-1}))
 yy=pd.concat(y); q=x[['date','symbol','f']].merge(yy,on=['date','symbol']).dropna(); aa=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.frh.nunique()>1: aa.append(g.f.corr(g.frh))
 print('horizon',h,'IC',np.mean(aa),'n_dates',len(aa))
