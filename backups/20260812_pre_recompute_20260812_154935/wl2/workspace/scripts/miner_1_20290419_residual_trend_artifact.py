import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 x=get_stock_daily_data(s,days=2400)
 if x is not None:D[s]=x.sort_values('date').set_index('date').close.astype(float)
c=sorted(set.intersection(*[set(D[s].index) for s in A]));P=pd.DataFrame({s:D[s].reindex(c) for s in A},index=c).ffill();R=P.pct_change();M=R.mean(axis=1); rows=[]
for i in range(62,len(P)-1):
 vals={}
 for s in A:
  a=R[s].iloc[i-59:i+1];b=M.iloc[i-59:i+1];beta=a.cov(b)/(b.var()+1e-10);e=a-beta*b;breadth=(R.iloc[i-9:i+1]>0).mean(axis=1).mean();vals[s]=e.iloc[-15:].sum()/(e.iloc[-20:].std()+1e-6)*(1-.5*abs(2*breadth-1))
 for s,v in vals.items():rows.append({'date':P.index[i].date().isoformat(),'symbol':s,'signal':float(v)})
pd.DataFrame(rows).to_csv('scripts/miner_1_20290419_residual_trend_signal.csv',index=False)
print('artifact rows',len(rows),'dates',len(set(x['date'] for x in rows)))
