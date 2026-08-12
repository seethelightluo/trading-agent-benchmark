import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(1); disp=R.std(1); gate=disp.rolling(20).mean()
rows=[]; sig=[]
# Shrink 60d beta toward 1; residual reversal remains volatility scaled and dispersion gated.
for t in range(85,len(P)-1):
 g=disp.iloc[t]/gate.iloc[t]
 if not np.isfinite(g): continue
 v={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<20 or z.iloc[:,1].var()<=1e-12: continue
  b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); b=.5*b+.5
  vol=z.iloc[:,0].std(); res=(R[s].iloc[t-2:t+1]-b*m.iloc[t-2:t+1]).sum()
  if vol>1e-8: v[s]=-res/vol*min(2,max(0,float(g)))
 q=pd.concat([pd.Series(v),R.iloc[t+1].reindex(v)],axis=1).dropna()
 if len(q)>=8:
  rows.append((P.index[t],len(q),q.iloc[:,0].corr(q.iloc[:,1]))); sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=o.ic
print('assets',len(P.columns),'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for c in ['2028-01-01','2029-01-01','2029-07-01']:
 b=a[a.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1))
pd.DataFrame(sig).to_csv('scripts/miner_3_20291004_shrunk_beta_residual3_signal.csv',index_label='date')
