import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
def avg(names,t,w):
 z=R.loc[:,R.columns.intersection(names)].iloc[t-w+1:t+1].mean(axis=1).sum()
 return z if np.isfinite(z) else np.nan
def evaluate(name,fn):
 rows=[]; signals=[]
 for t in range(65,len(P)-11):
  v=fn(t)
  for h in (1,5,10):
   q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
   if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
  signals.append(pd.Series(v,name=P.index[t]))
 o=pd.DataFrame(rows,columns=['date','h','n','ic'])
 print('FACTOR',name,'universe',len(U),'dates',len(P))
 for h in (1,5,10):
  z=o[o.h==h].set_index('date').ic
  print('h',h,'dates',len(z),'avgN',o[o.h==h].n.mean(),'coverage',o[o.h==h].n.mean()/len(U),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
  for c in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
   b=z[z.index>=c]; print(c,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
 S=pd.DataFrame(signals); S.to_csv('scripts/miner_2_20291115_'+name+'_signal.csv',index_label='date')
 print('turnover',np.nanmean((S.diff().abs().sum(1)/(S.abs().sum(1)+1e-9)).values))
def factor(t):
 defs=['XAU','US10Y','CN10Y']; risks=['SOX','NDX','BTC','ETH','WTI']
 lead=avg(defs,t,20)-avg(risks,t,20)
 v={}
 for s in P:
  x=R[s].iloc[t-59:t+1].dropna()
  if len(x)>=45 and x.std()>1e-8:
   excess=R[s].iloc[t-9:t+1].sum()-R.iloc[t-9:t+1].mean(axis=1).sum()
   v[s]=excess/(x.std()*np.sqrt(10))*(1+np.tanh(lead/.08))
 return v
evaluate('defensive_lead_relative_strength',factor)
