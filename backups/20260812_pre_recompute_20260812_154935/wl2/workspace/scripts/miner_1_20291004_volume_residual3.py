import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={};vol={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:
  x=d.set_index('date');cl[s]=x.close.astype(float);vol[s]=x.volume.astype(float)
P=pd.DataFrame(cl).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change(); m=R.mean(1); disp=R.std(1)
# Volume-weighted residual reversal: reversal is stronger when recent volume is elevated,
# but capped to limit noisy crypto/market-volume scale differences.
rows=[]; sig=[]
for t in range(100,len(P)-1):
 vals={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<40 or z.iloc[:,1].var()<=1e-12: continue
  b=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vv=z.iloc[:,0].std()
  rr=(R[s].iloc[t-2:t+1]-b*m.iloc[t-2:t+1]).sum()
  vr=V[s].iloc[t-19:t+1].replace([np.inf,-np.inf],np.nan).dropna()
  if vv<=1e-8 or len(vr)<10 or vr.mean()<=0: continue
  activity=np.clip(np.log1p(V[s].iloc[t]/vr.mean()),-1,1)
  vals[s]=-rr/vv*(1+0.35*activity)
 q=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals)],axis=1).dropna()
 if len(q)>=8: rows.append((P.index[t],len(q),q.iloc[:,0].corr(q.iloc[:,1])));sig.append(pd.Series(vals,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=o.ic
print('idea=volume_amplified_residual_reversal_3d assets',len(P.columns),'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.mean()/len(U))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for c in ['2026-07-16','2028-01-01','2029-01-01','2029-07-01']:
 b=a[a.index>=c]; print(c,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(sig);print('turnover_proxy',S.diff().abs().mean().mean());S.to_csv('scripts/miner_1_20291004_volume_residual3_signal.csv',index_label='date')
