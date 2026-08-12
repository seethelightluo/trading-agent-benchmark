import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change();m=R.mean(axis=1)
for bw,lh in [(10,3),(10,5),(40,5),(40,3)]:
 rows=[]
 for t in range(bw+lh,len(P)-1):
  vals={}
  for s in P:
   z=pd.concat([R[s].iloc[t-bw+1:t+1],m.iloc[t-bw+1:t+1]],axis=1).dropna()
   if len(z)<max(8,bw//2):continue
   beta=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/np.var(z.iloc[:,1],ddof=1)
   vol=R[s].iloc[t-bw+1:t+1].std()
   vals[s]=-((R[s].iloc[t-lh+1:t+1]-beta*m.iloc[t-lh+1:t+1]).sum())/(vol*np.sqrt(bw)) if vol>1e-8 else np.nan
  f=pd.Series(vals).dropna();q=pd.concat([f,R.iloc[t+1].reindex(f.index)],axis=1).dropna()
  if len(q)>=8:rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q),f))
 a=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows]); print('bw/lh',bw,lh,'dates',len(a),'avgN',ns.mean(),'cov',ns.mean()/len(P.columns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 for lab,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
  b=a[[x[0]>=cut for x in rows]];print(lab,'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)))
