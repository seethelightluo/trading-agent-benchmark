import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; signals=[]
# Reversal of directional efficiency: noisy/persistent 20d moves are
# measured by net return / sum absolute returns and inverted. Scale by
# inverse volatility to prefer cleaner, less risky rebound candidates.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  r=R[s].iloc[t-19:t+1].dropna()
  if len(r)<18: continue
  eff=r.sum()/(r.abs().sum()+1e-12)
  vol=R[s].iloc[t-59:t+1].std()
  if not np.isfinite(vol) or vol<=1e-8: continue
  v[s]=-eff/(vol*np.sqrt(20))
 for h in (1,5,10):
  fwd=R.iloc[t+1:t+h+1].sum().reindex(v)
  q=pd.concat([pd.Series(v),fwd],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 signals.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(o[o.h==h].n.mean(),3),'coverage',round(o[o.h==h].n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(signals); S.to_csv('scripts/miner_3_20291129_efficiency_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values))
