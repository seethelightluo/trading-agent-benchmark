import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100:d=get_index_daily_data(s,1800)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change(); rows=[]; sig=[]
# Conditional shock reversal: reverse the latest 3-day move, but only when
# the asset's 20-day trend agrees with the opposite direction. This avoids
# blindly fading persistent trends and uses volatility normalization.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  r=R[s].iloc[t-59:t+1].dropna()
  if len(r)<50:continue
  vol=r.iloc[-20:].std()
  if not np.isfinite(vol) or vol<1e-8:continue
  shock=r.iloc[-3:].sum(); trend=r.iloc[-20:].sum()
  # fade shocks only if medium trend is not in the shock direction
  if shock*trend>0: score=0.0
  else: score=-shock/(vol*np.sqrt(3))
  v[s]=score
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8:rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h];a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c];print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
pd.DataFrame(sig).to_csv('scripts/miner_3_20300207_conditional_shock_signal.csv',index_label='date')
print('signal_rows',len(sig),'instruments',len(U),'available',len(px))
