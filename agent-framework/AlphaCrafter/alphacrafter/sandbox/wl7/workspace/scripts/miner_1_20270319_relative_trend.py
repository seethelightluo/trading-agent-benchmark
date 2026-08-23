import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=get_account_dict()['watch_list']; frames={}
for s in U:
    try: d=get_index_daily_data(s, days=2600)
    except Exception: d=None
    if d is None or len(d)<150:
        try: d=get_stock_daily_data(s, days=2600)
        except Exception: d=None
    if d is not None:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
p=pd.concat(frames,axis=1).sort_index().ffill(); r=p.pct_change()
mom=p.pct_change(20); med=mom.median(axis=1); vol=r.rolling(30).std()*np.sqrt(20); f=(mom.sub(med,axis=0)/vol).shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('h',h,'dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().sum().sum()/f.size,'rank_turnover',rank.diff().abs().mean(axis=1).mean(),'period',p.index.min(),p.index.max(),'assets',p.shape[1])
fr=p.shift(-1)/p-1
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(a,b,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270319_relative_trend_signal.csv',index=False)
