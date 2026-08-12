import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is None or len(d)<200: d=get_index_daily_data(s,2500)
 if d is not None:
  x=d.set_index('date'); px[s]=x.close.astype(float); vol[s]=x.volume.astype(float) if 'volume' in x else pd.Series(index=x.index,dtype=float)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change()
rows=[]; turnover=[]; prev=None; valid_dates=0
# Clean volume-shock reversal: negative 3d return, scaled by log-volume surprise over a 60d median.
for t in range(65,len(P)-11):
 f={}
 for s in P.columns:
  rr=R[s]; vv=V[s]
  if t<3 or vv.iloc[t-3:t+1].isna().any(): continue
  base=vv.iloc[t-63:t-3].replace([np.inf,-np.inf],np.nan).dropna()
  if len(base)<30 or base.median()<=0: continue
  recent=vv.iloc[t-2:t+1].replace([np.inf,-np.inf],np.nan).dropna()
  if len(recent)<2: continue
  shock=np.log((recent.median()+1e-12)/(base.median()+1e-12))
  ret=rr.iloc[t-2:t+1].sum()
  if np.isfinite(ret) and np.isfinite(shock): f[s]=float(-ret*max(shock,0.0))
 if len(f)>=8:
  valid_dates+=1
  cur=pd.Series(f).rank(pct=True)
  if prev is not None:
   ix=cur.index.intersection(prev.index)
   if len(ix): turnover.append((cur[ix]-prev[ix]).abs().mean())
  prev=cur
  for h in (1,5,10):
   fr=R.iloc[t+1:t+h+1].sum().reindex(cur.index)
   q=pd.concat([cur,fr],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
   if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('universe',len(P.columns),'dates',len(P),'valid_signal_dates',valid_dates)
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic.replace([np.inf,-np.inf],np.nan).dropna()
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),2),'coverage',round(z.n.sum()/len(z)/len(P.columns),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('turnover',round(float(np.mean(turnover)),6) if turnover else 'NA')
for cutoff in ['2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
 z=o[(o.h==1)&(o.date>=cutoff)].ic.replace([np.inf,-np.inf],np.nan).dropna()
 print('recent',cutoff,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
# signal artifact for reproducibility
out=[]
for t in range(65,len(P)-11):
 f={}
 for s in P.columns:
  vv=V[s]; rr=R[s]
  base=vv.iloc[t-63:t-3].replace([np.inf,-np.inf],np.nan).dropna()
  recent=vv.iloc[t-2:t+1].replace([np.inf,-np.inf],np.nan).dropna()
  if len(base)>=30 and base.median()>0 and len(recent)>=2:
   ret=rr.iloc[t-2:t+1].sum(); shock=np.log((recent.median()+1e-12)/(base.median()+1e-12))
   if np.isfinite(ret) and np.isfinite(shock): out.append({'date':P.index[t],'symbol':s,'signal':float(-ret*max(shock,0.0))})
pd.DataFrame(out).to_csv('scripts/miner_2_20300221_volume_shock_reversal_signal.csv',index=False)
