import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.sort_values('date'); z.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'symbol':s,'r3':d.close/d.close.shift(3)-1,'fr':d.close.shift(-1)/d.close-1}))
x=pd.concat(z); x['f0']=-x.r3
# cross-sectional percentile contrarian signal, lagged by construction via r3
x['f']=x.groupby('date').f0.rank(pct=True,method='average')
x[['date','symbol','f']].dropna().to_csv('scripts/miner_1_20271202_rank_reversal_signal.csv',index=False)
vals=[];ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.f.nunique()>1: vals.append(g.f.corr(g.fr));ns.append(len(g))
a=pd.Series(vals).dropna();print('dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',np.mean(np.array(ns)/15),'recent500',a.tail(500).mean())
