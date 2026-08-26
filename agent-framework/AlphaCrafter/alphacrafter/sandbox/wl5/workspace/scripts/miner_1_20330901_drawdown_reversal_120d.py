import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=get_account_dict()['watch_list']
D={}
for s in U:
 
 try: d=get_index_daily_data(s, days=5000)
 except Exception: d=None
 if d is None or len(d)<160:
  try: d=get_stock_daily_data(s, days=5000)
  except Exception: d=None
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
# signal at t uses close through t; forward 10-session return t+10/t
rows=[]
for i in range(119,len(p)-10):
 x=p.iloc[i]; f=1-x/p.iloc[i-119:i+1].max(); y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
 if len(z)>=8:
  rows.append((p.index[i],z['f'].corr(z['y'],method='spearman'),len(z),z['f'].rank().corr(z['y'].rank(),method='pearson')))
r=pd.DataFrame(rows,columns=['date','ic','n','rankic']).set_index('date')
# rankic is same, use it
q=r.ic.dropna(); mean=q.mean(); sd=q.std(ddof=1)
print('factor=120d_peak_drawdown_reversal dates',len(q),'mean_n',r.n.mean(),'coverage',r.n.sum()/(len(q)*len(U)))
print('IC',mean,'ICIR',mean/sd*np.sqrt(252) if sd else np.nan,'hit', (q>0).mean(),'turnover',np.nan)
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2033-08-31')]:
 v=q.loc[a:b]
 print(a,b,len(v),v.mean(),v.mean()/v.std(ddof=1)*np.sqrt(252) if len(v)>1 else np.nan)
print('period',q.index.min(),q.index.max())
# artifact
out=[]
for i in range(119,len(p)-10):
 f=1-p.iloc[i]/p.iloc[i-119:i+1].max()
 for s in U:
  if s in f.index and np.isfinite(f[s]): out.append({'date':p.index[i].date().isoformat(),'symbol':s,'signal':float(f[s])})
pd.DataFrame(out).to_csv('scripts/miner_1_20330901_drawdown_reversal_120d_signal.csv',index=False)
