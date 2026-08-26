import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-10-17'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
R={}; V={}
for s,x in D.items():
 c=x.close.astype(float); R[s]=c.pct_change(20); V[s]=c.pct_change().rolling(40,min_periods=25).std()
R=pd.DataFrame(R); V=pd.DataFrame(V); breadth=(R>0).mean(axis=1); rows=[]
for s,x in D.items():
 idx=x.index; r=R[s].reindex(idx); v=V[s].reindex(idx); br=breadth.reindex(idx)
 f=(-(r-R.mean(axis=1).reindex(idx))/v.replace(0,np.nan))*np.where(br<0.5,1.25,0.65)
 for h in [1,5,10,20]: rows.append(pd.DataFrame({'date':idx,'factor':f.to_numpy(),'fwd':(x.close.shift(-h)/x.close-1).to_numpy(),'h':h,'s':s}))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
for h,g0 in a.groupby('h'):
 z=[]; ns=[]
 for d,g in g0.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>2 and g.fwd.nunique()>2:z.append(g.factor.corr(g.fwd)); ns.append(len(g))
 v=pd.Series(z).dropna(); print('H',h,'dates',len(v),'avgN',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean(),'thirds',[v.iloc[i*len(v)//3:(i+1)*len(v)//3].mean() for i in range(3)])
 if h==10: g0[['date','s','factor']].to_csv('scripts/miner_3_20321018_breadth_residual_signal.csv',index=False)
print('instruments',len(D),'coverage',len(a)/sum(len(x) for x in D.values()))
