import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,1200)
 if d is None or len(d)<100:d=get_index_daily_data(s,1200)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change();m=R.mean(1);disp=R.std(1);ds=disp.rolling(20).mean();rows=[];sig=[]
for t in range(85,len(P)-1):
 x=disp.iloc[t]/ds.iloc[t]
 if not np.isfinite(x):continue
 v={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<20 or z.iloc[:,1].var()<=1e-12:continue
  b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std();res=(R[s].iloc[t-4:t+1]-b*m.iloc[t-4:t+1]).sum()
  if vol>1e-8:v[s]=-res/vol*min(2,max(0,float(x)))
 q=pd.concat([pd.Series(v),R.iloc[t+1].reindex(v)],axis=1).dropna()
 if len(q)>=8:rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q)));sig.append(pd.Series(v,name=P.index[t]))
a=np.array([r[1] for r in rows]);n=np.array([r[2] for r in rows]);print('dates',len(a),'assets',len(P.columns),'avgN',n.mean(),'coverage',n.mean()/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0));
for cut in ['2028-01-01','2029-01-01']:
 b=a[[r[0]>=pd.Timestamp(cut) for r in rows]];print(cut,len(b),b.mean(),b.mean()/b.std(ddof=1))
pd.DataFrame(sig).to_csv('scripts/miner_3_20290823_dispersion_residual5_signal.csv',index_label='date')
