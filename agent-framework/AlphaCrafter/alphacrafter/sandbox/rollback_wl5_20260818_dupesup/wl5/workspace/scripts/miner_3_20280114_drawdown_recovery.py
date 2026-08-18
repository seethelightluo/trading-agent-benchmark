import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# causal factor: recovery pressure = negative drawdown from 60d high, adjusted by 20d vol;
# higher means deeper drawdown (cross-sectional relative), hypothesized rebound
px={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d)>100: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
rets=P.pct_change()
# align dates with enough names and forward returns
rows=[]
for i in range(70,len(P)-10):
 date=P.index[i]
 vals={}; fw={}
 for s in P.columns:
  x=P[s].iloc[:i+1]
  if len(x.dropna())<65 or pd.isna(P[s].iloc[i+10]): continue
  vol=rets[s].iloc[i-19:i+1].std()
  if not np.isfinite(vol) or vol<=0: continue
  dd=P[s].iloc[i]/P[s].iloc[i-59:i+1].max()-1
  vals[s]=(-dd)/vol
  fw[s]=P[s].iloc[i+10]/P[s].iloc[i]-1
 if len(vals)>=8:
  a=pd.Series(vals); b=pd.Series(fw).reindex(a.index)
  rows.append((date,a.corr(b),len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'instruments_mean',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for h in [1,3,5,10]:
 rows=[]
 for i in range(70,len(P)-h):
  vals={}; fw={}
  for s in P.columns:
   vol=rets[s].iloc[i-19:i+1].std(); x=P[s].iloc[:i+1]
   if len(x.dropna())<65 or not np.isfinite(vol) or vol<=0 or pd.isna(P[s].iloc[i+h]): continue
   vals[s]=-(P[s].iloc[i]/P[s].iloc[i-59:i+1].max()-1)/vol; fw[s]=P[s].iloc[i+h]/P[s].iloc[i]-1
  if len(vals)>=8: rows.append(pd.Series(vals).corr(pd.Series(fw).reindex(vals.keys())))
 q=pd.Series(rows).dropna(); print('horizon',h,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
# signal artifact for admission horizon 10
out=[]
for i in range(70,len(P)-10):
 vals={}; fw={}
 for s in P.columns:
  vol=rets[s].iloc[i-19:i+1].std(); x=P[s].iloc[:i+1]
  if len(x.dropna())<65 or not np.isfinite(vol) or vol<=0 or pd.isna(P[s].iloc[i+10]): continue
  vals[s]=-(P[s].iloc[i]/P[s].iloc[i-59:i+1].max()-1)/vol; fw[s]=P[s].iloc[i+10]/P[s].iloc[i]-1
 if len(vals)>=8:
  for s,v in vals.items(): out.append({'date':P.index[i],'symbol':s,'signal':v,'forward_return_10d':fw[s]})
pd.DataFrame(out).to_csv('scripts/miner_3_20280114_drawdown_recovery_signal.csv',index=False)
