import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
for t in range(45,len(P)-1):
 v={}
 for s in P:
  x=R[s].iloc[t-39:t+1].dropna()
  if len(x)<30: continue
  a=np.sign(R[s].iloc[t-29:t+1].sum())*R[s].iloc[t-9:t+1].sum()/(x.std()+1e-6)
  v[s]=a
 q=pd.concat([pd.Series(v),R.iloc[t+1].reindex(v)],axis=1).dropna()
 if len(q)>=8: rows.append((P.index[t],len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=o.ic
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for c in ['2028-01-01','2029-01-01','2029-07-01','2029-10-01']:
 b=a[a.index>=c];print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
