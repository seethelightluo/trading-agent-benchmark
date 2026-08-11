import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is None: continue
 d=d.sort_values('date').copy(); d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
 # lagged intraday pressure: prior completed session open-to-close, contrarian
 d['f']=-(d.close/d.open-1).shift(1)
 for h in [1,3,5,10]: d[f'fr{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','f']+[f'fr{h}' for h in [1,3,5,10]]].assign(symbol=s))
x=pd.concat(rows); x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20271202_intraday_reversal_signal.csv',index=False)
for h in [1,3,5,10]:
 vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g[['f',f'fr{h}']].dropna()
  if len(g)>=8 and g.f.nunique()>1:
   vals.append(g.f.corr(g[f'fr{h}'])); ns.append(len(g))
 a=pd.Series(vals).dropna(); print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),4),'hit',round((a>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4),'recent500',round(a.tail(500).mean(),6))
print('turnover proxy', x.sort_values(['symbol','date']).groupby('symbol').f.apply(lambda s: (s.rank(pct=True).diff().abs()>0.1).mean()).mean())
