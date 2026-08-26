import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=get_account_dict()['watch_list']; D={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,days=5000)
 except:pass
 if d is None:
  try:d=get_stock_daily_data(s,days=5000)
  except:pass
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); rows=[]; out=[]
for i in range(59,len(p)-10):
 win=p.iloc[i-59:i+1]; lo=win.min(); hi=win.max(); den=(hi-lo).replace(0,np.nan)
 f=(1-(p.iloc[i]-lo)/den).replace([np.inf,-np.inf],np.nan)
 y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],z.f.corr(z.y,method='spearman'),len(z)))
 for s in U:
  if s in f.index and np.isfinite(f[s]):out.append({'date':p.index[i].date().isoformat(),'symbol':s,'signal':float(f[s])})
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=r.ic.dropna(); m=q.mean(); sd=q.std(ddof=1)
print('factor=range_position_reversal_60d dates',len(q),'mean_n',r.n.mean(),'coverage',r.n.sum()/(len(q)*len(U)))
print('IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(q>0).mean(),'period',q.index.min(),q.index.max())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2033-08-31')]:
 v=q.loc[a:b]; print(a,b,len(v),v.mean(),v.mean()/v.std(ddof=1)*np.sqrt(252) if len(v)>1 else np.nan)
pd.DataFrame(out).to_csv('scripts/miner_1_20330915_range_position_reversal_60d_signal.csv',index=False)
