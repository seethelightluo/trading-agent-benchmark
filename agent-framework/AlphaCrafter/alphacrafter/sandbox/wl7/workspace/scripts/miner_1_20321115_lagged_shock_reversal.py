import numpy as np,pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-11-14'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
P=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index(); R=P.pct_change()
rows=[]
for s,x in D.items():
 c=x.close.astype(float); r=R[s].reindex(c.index)
 # Lagged 3-session shock reversal, normalized by realized volatility; signal known at t
 f=-r.rolling(3,min_periods=3).sum().shift(3)/r.rolling(30,min_periods=20).std().shift(1)
 for h in [1,5,10,20]:
  rows.append(pd.DataFrame({'date':c.index,'symbol':s,'factor':f,'fwd':c.shift(-h)/c-1,'h':h}))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
for h in [1,5,10,20]:
 q=a[a.h==h]; out=[]; ns=[]
 for d,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3: out.append(g.factor.corr(g.fwd)); ns.append(len(g))
 v=pd.Series(out).dropna(); print('H',h,'dates',len(v),'avgN',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),4))
# coverage and simple rank turnover
q=a[a.h==10]; print('overall dates',P.index.min().date(),P.index.max().date(),'assets',len(D),'coverage',round(len(q)/(len(D)*len(P.index)),4))
# thirds
out=[]
for d,g in q.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>=3: out.append((d,g.factor.corr(g.fwd)))
ic=pd.DataFrame(out,columns=['date','ic']); n=len(ic)//3
print('thirds',[round(ic.iloc[i* n:(i+1)*n].ic.mean(),6) for i in range(3)])
os.makedirs('scripts',exist_ok=True); q[['date','symbol','factor']].to_csv('scripts/miner_1_20321115_lagged_shock_reversal_signal.csv',index=False)
