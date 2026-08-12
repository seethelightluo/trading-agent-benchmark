import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2200)
 if d is None or len(d)<100: d=get_index_daily_data(s,2200)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Candidate: dispersion-gated 3-day residual reversal. Reversal is activated when
# cross-sectional 3d dispersion is high, where dislocations tend to mean-revert.
rows=[]; sig=[]
for t in range(70,len(P)-11):
 r3=R.iloc[t-2:t+1].sum(); med=r3.median(); disp=r3.std()
 vol=R.iloc[t-29:t+1].std()
 if not np.isfinite(disp) or disp<=0: continue
 gate=np.clip(disp/(R.iloc[t-59:t+1].std().mean()+1e-9),0.5,2.0)
 f=(-(r3-med)/(vol+1e-9)*gate).dropna()
 y=R.iloc[t+1].reindex(f.index)
 q=pd.concat([f,y],axis=1).dropna()
 if len(q)>=8: rows.append((P.index[t],len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(f.rename(P.index[t]))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); z=o.ic
print('FACTOR dispersion_gated_3d_residual_reversal','universe',len(U),'data_dates',len(P),'ic_dates',len(z),'avgN',o.n.mean(),'coverage',o.n.mean()/len(U))
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for h in (3,5,10):
 rr=[]
 for t in range(70,len(P)-h):
  r3=R.iloc[t-2:t+1].sum(); med=r3.median(); disp=r3.std(); vol=R.iloc[t-29:t+1].std(); gate=np.clip(disp/(R.iloc[t-59:t+1].std().mean()+1e-9),.5,2.)
  f=(-(r3-med)/(vol+1e-9)*gate).dropna(); y=R.iloc[t+1:t+h+1].sum().reindex(f.index); q=pd.concat([f,y],axis=1).dropna()
  if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1]))
 print('horizon',h,'dates',len(rr),'IC',np.nanmean(rr))
S=pd.concat(sig,axis=1).T if sig else pd.DataFrame(); S.to_csv('scripts/miner_2_20310123_dispersion_gated_3d_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean(S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031')]:
 zz=z[(z.index>=a)&(z.index<=b)]; print('regime',a,b,'dates',len(zz),'IC',zz.mean(),'ICIR',zz.mean()/zz.std(ddof=1) if len(zz)>1 else np.nan)
