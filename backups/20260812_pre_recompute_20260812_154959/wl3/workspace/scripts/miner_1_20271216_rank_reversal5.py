import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is None:continue
 d=d.sort_values('date'); c=d.close.astype(float)
 rows.append(pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'symbol':s,'raw':-(c/c.shift(5)-1),'fr':c.shift(-1)/c-1}))
x=pd.concat(rows,ignore_index=True); x['f']=x.groupby('date').raw.rank(pct=True,method='average'); out=x[['date','symbol','f']].dropna(); out.to_csv('scripts/miner_1_20271216_rank_reversal5_signal.csv',index=False)
a=[];n=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fr'])
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr));n.append(len(g))
a=pd.Series(a).dropna();p=out.pivot(index='date',columns='symbol',values='f').sort_index();to=(p.diff().abs().mean(axis=1)/2).mean()
print('dates',len(a),'avg_n',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'coverage',np.mean(np.array(n)/15),'turnover_proxy',to,'recent500',a.tail(500).mean())
