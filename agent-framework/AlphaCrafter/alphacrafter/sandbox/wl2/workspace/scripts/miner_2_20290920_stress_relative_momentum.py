import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<200: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); eq=[x for x in ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX'] if x in P]
# Observation-only VIX is used solely as a lagged stress regime input.
v=get_index_daily_data('VIX',1800)
V=v.set_index('date').close.astype(float) if v is not None else pd.Series(index=P.index,dtype=float)
V=V.reindex(P.index).ffill(); eqret=P[eq].pct_change(20).mean(axis=1); breadth=(P[eq].pct_change(20)>0).mean(axis=1)
variants=[(20,60,0.5,False),(20,60,0.5,True),(40,90,0.5,False),(20,120,0.4,False)]
for win,reg,br,rev in variants:
 rows=[]; turns=[]; prev=None; ns=[]
 for t in range(max(win,reg,25),len(P)-1):
  # regime known at t, signal predicts t+1; stress if VIX elevated or equity breadth weak
  vh=V.iloc[t-reg+1:t+1].dropna()
  if len(vh)<reg*.8 or not np.isfinite(breadth.iloc[t]): continue
  stress=(V.iloc[t]>=vh.median()) or (breadth.iloc[t]<=br)
  vals={}
  for s in P:
   r=R[s].iloc[t-win+1:t+1].sum(); vol=R[s].iloc[t-win+1:t+1].std()
   if np.isfinite(r) and vol>1e-8: vals[s]=(r/vol)*(1 if stress else 0)
  if rev: vals={s:-x for s,x in vals.items()}
  q=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals.keys())],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].std()>0:
   ic=q.iloc[:,0].corr(q.iloc[:,1])
   if np.isfinite(ic): rows.append((P.index[t],ic,len(q)))
  if prev is not None: turns.append((pd.Series(vals).reindex(P.columns).fillna(0)-prev.reindex(P.columns).fillna(0)).abs().mean())
  prev=pd.Series(vals); ns.append(len(q))
 a=pd.Series([x[1] for x in rows]);
 print('VAR',win,reg,br,'reverse',rev,'dates',len(a),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f turnover %.4f coverage %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(turns),np.mean(ns)/len(P.columns)))
 for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
  z=a[[x[0]>=cut for x in rows]]; print(lab,len(z),'%.6f %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
