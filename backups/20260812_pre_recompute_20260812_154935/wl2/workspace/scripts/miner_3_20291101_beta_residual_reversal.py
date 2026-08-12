import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# Short-horizon beta-residual reversal: remove each asset's 60d beta to
# equal-weight universe, then invert the recent 3d residual move. Volatility
# normalization and dispersion gating reduce unstable cross-sections.
M=R.mean(axis=1)
for t in range(90,len(P)-11):
 v={}
 disp=R.iloc[t-19:t+1].std(axis=1).mean()
 for s in P:
  a=R[s].iloc[t-59:t+1]; b=M.iloc[t-59:t+1]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)<45 or z.iloc[:,1].var()<1e-10: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var()
  resid=(R[s]-beta*M).iloc[t-2:t+1].sum()
  vol=a.std()
  v[s]=-resid/(vol*np.sqrt(3)+1e-9) * np.clip(disp/0.012,0.5,2.0)
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for c in ['2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20291101_beta_residual_reversal_signal.csv',index_label='date')
print('turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values))
