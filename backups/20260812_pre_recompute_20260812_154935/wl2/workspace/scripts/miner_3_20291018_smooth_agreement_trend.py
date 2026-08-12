import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
rows=[]; sig=[]
# Low-turnover risk-agreement trend: 20d return, aligned with 40d trend, scaled by 60d volatility.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  x=R[s].iloc[t-59:t+1].dropna()
  if len(x)<45: continue
  vol=x.std()
  if vol>1e-8:
   v[s]=np.sign(R[s].iloc[t-39:t+1].sum())*R[s].iloc[t-19:t+1].sum()/vol
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic']);
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic; print('h',h,'dates',len(a),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for c in ['2028-01-01','2029-01-01','2029-07-01','2029-10-01']:
  b=a[a.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
a=o[o.h==1].set_index('date').ic
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20291018_smooth_agreement_trend_signal.csv',index_label='date')
print('signal rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(1)/(S.abs().sum(1)+1e-9)).values))
