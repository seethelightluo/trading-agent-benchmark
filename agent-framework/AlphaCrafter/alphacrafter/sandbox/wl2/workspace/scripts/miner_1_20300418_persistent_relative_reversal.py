import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2400)
 if d is None or len(d)<120: d=get_index_daily_data(s,2400)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# Novel candidate: short-horizon residual reversal with a persistence filter.
# Fade the asset's 3-day return relative to the contemporaneous universe,
# but only when the 3-day relative move agrees with the 10-day relative move.
# This avoids fading ordinary trend and targets overshoot after persistent divergence.
for t in range(65,len(P)-11):
 r3=R.iloc[t-2:t+1].sum(); r10=R.iloc[t-9:t+1].sum()
 m3=r3.mean(); m10=r10.mean(); disp=R.iloc[t-19:t+1].sum().std()
 if not np.isfinite(disp) or disp<=1e-9: continue
 f={}
 for s in P:
  v=R[s].iloc[t-19:t+1].std(ddof=1)
  if np.isfinite(v) and v>1e-9 and np.isfinite(r3[s]) and np.isfinite(r10[s]):
   rel3=r3[s]-m3; rel10=r10[s]-m10
   if rel3*rel10>0: f[s]=-rel3/v
 f=pd.Series(f); sig.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index); q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_1_20300418_persistent_relative_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'instruments',len(U),'available',len(px))
