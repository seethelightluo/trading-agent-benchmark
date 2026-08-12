import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
for t in range(65,len(P)-11):
 raw={}
 for s in P:
  r=R[s].iloc[t-59:t+1].dropna()
  if len(r)>=45 and r.std()>1e-8:
   ret=r.sum(); breadth=(r>0).mean(); raw[s]=-(ret/(r.std()*np.sqrt(60)))*breadth
 v=pd.Series(raw).rank(pct=True).to_dict()
 for h in (1,5,10,20):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10,20):
 z=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(z),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 for c in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=z[z.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_2_20291115_rank_reversal_signal.csv',index_label='date')
print('turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values))
