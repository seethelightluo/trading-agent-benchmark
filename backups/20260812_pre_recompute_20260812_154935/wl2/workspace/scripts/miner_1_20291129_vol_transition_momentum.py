import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
rows=[]; sig=[]
# Volatility-state transition momentum: medium-term trend is favored when
# current 10d volatility is contracting versus its 60d baseline; signals are
# risk-scaled and cross-asset comparable.
for t in range(70,len(P)-11):
 v={}
 for s in P:
  r=R[s]
  if r.iloc[t-59:t+1].count()<55: continue
  rv10=r.iloc[t-9:t+1].std(ddof=1); rv60=r.iloc[t-59:t+1].std(ddof=1)
  mom=r.iloc[t-19:t+1].sum()
  if np.isfinite(rv10) and np.isfinite(rv60) and rv60>1e-7:
   # positive trend during vol compression; negative trend during expansion
   v[s]=mom*(1-(rv10/rv60-1))*0.5/(rv60*np.sqrt(252))
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for c in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_1_20291129_vol_transition_momentum_signal.csv',index_label='date')
print('turnover',np.nanmean((S.diff().abs().sum(1)/(S.abs().sum(1)+1e-9)).values))
