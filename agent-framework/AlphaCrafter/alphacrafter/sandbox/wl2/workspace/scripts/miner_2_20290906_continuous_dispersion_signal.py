import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1600)
 if d is None or len(d)<150: d=get_index_daily_data(s,1600)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1); disp=R.std(axis=1)
bw,lh,qwin,q,cap=60,3,120,.65,2.0
rows=[]; signals=[]; turns=[]; prev=None
for t in range(max(bw+lh,qwin),len(P)-1):
 h=disp.iloc[t-qwin+1:t+1].dropna()
 if len(h)<qwin*.8 or not np.isfinite(disp.iloc[t]): continue
 intensity=max(0.,min(cap,disp.iloc[t]/h.median()-1.0)) if disp.iloc[t]>=h.quantile(q) else 0.
 vals={}
 for s in P:
  z=pd.concat([R[s].iloc[t-bw+1:t+1],m.iloc[t-bw+1:t+1]],axis=1).dropna()
  if len(z)<20 or z.iloc[:,1].var()<=1e-12: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var(); vol=z.iloc[:,0].std()
  resid=(R[s].iloc[t-lh+1:t+1]-beta*m.iloc[t-lh+1:t+1]).sum()
  if vol>1e-8: vals[s]=-resid/vol*intensity
 qx=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals.keys())],axis=1).dropna()
 if len(qx)>=8 and qx.iloc[:,0].std()>0:
  ic=qx.iloc[:,0].corr(qx.iloc[:,1])
  if np.isfinite(ic): rows.append((P.index[t],ic,len(qx))); signals.append(pd.Series(vals,name=P.index[t]))
 if prev is not None: turns.append((pd.Series(vals).reindex(P.columns).fillna(0)-prev.reindex(P.columns).fillna(0)).abs().mean())
 prev=pd.Series(vals)
a=pd.Series([x[1] for x in rows]); out=pd.DataFrame(signals); out.to_csv('scripts/miner_2_20290906_continuous_dispersion_signal.csv',index_label='date')
print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'IC %.6f ICIR %.6f turnover %.6f coverage %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(turns),np.mean([x[2] for x in rows])/len(P.columns)))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 z=a[[x[0]>=cut for x in rows]]; print(lab,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
