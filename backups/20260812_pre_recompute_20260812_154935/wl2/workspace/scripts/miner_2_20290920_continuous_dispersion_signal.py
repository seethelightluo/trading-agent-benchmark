import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<200: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1); disp=R.std(axis=1); rows=[]
for t in range(120+60,len(P)-1):
 h=disp.iloc[t-119:t+1].dropna()
 if len(h)<96 or not np.isfinite(disp.iloc[t]): continue
 intensity=max(0.,min(2.,disp.iloc[t]/h.median()-1)) if disp.iloc[t]>=h.quantile(.65) else 0.
 vals={}
 for s in P:
  z=pd.concat([R[s].iloc[t-59:t+1],m.iloc[t-59:t+1]],axis=1).dropna()
  if len(z)<20 or z.iloc[:,1].var()<=1e-12: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std()
  resid=(R[s].iloc[t-2:t+1]-beta*m.iloc[t-2:t+1]).sum()
  if vol>1e-8: vals[s]=-resid/vol*intensity
  rows.append({'date':P.index[t], 'symbol':s, 'signal':vals.get(s,np.nan), 'forward_return':R[s].iloc[t+1]})
pd.DataFrame(rows).to_csv('scripts/miner_2_20290920_continuous_dispersion_signal.csv',index=False)
print('saved',len(rows),'rows','dates',pd.DataFrame(rows).date.nunique())
