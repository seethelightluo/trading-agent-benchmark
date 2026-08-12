import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<100: d=get_index_daily_data(s,1800)
 if d is not None: cl[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(cl).sort_index(); R=P.pct_change(); rows=[]; signals=[]
# Conditional defensive volatility: inverse recent volatility, rewarded only when asset's trend is non-negative.
for t in range(60,len(P)-5):
 vals={}
 for s in P:
  r=R[s].iloc[t-19:t+1].dropna(); long=R[s].iloc[t-59:t+1].dropna()
  if len(r)<15 or len(long)<45: continue
  v=r.std(); trend=long.sum()
  if v<=1e-8: continue
  vals[s]=(1/v)*(0.5+0.5/(1+np.exp(-trend/0.05)))
 q=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals),R.iloc[t+5].reindex(vals)],axis=1).dropna()
 if len(q)>=8:
  rows.append((P.index[t],len(q),q.iloc[:,0].corr(q.iloc[:,1]),q.iloc[:,0].corr(q.iloc[:,2]))); signals.append(pd.Series(vals,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','n','ic1','ic5']).set_index('date')
print('assets',len(P.columns),'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.mean()/len(U))
for h in ['ic1','ic5']:
 a=o[h].dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for c in ['2026-07-16','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
S=pd.DataFrame(signals); print('turnover_proxy',S.diff().abs().mean().mean()); S.to_csv('scripts/miner_1_20291101_defensive_vol_signal.csv',index_label='date')
