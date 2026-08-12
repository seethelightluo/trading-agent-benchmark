import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2400)
 if d is None or len(d)<100: d=get_index_daily_data(s,2400)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# Conditional shock-reversal: fade 3d return, amplified by cross-asset stress
# (negative median 5d return and high cross-sectional 5d dispersion), plus volume surprise.
for t in range(65,len(P)-11):
 avail=P.iloc[:t+1]; rr=R.iloc[:t+1]
 m5=rr.iloc[t-4:t+1].mean(axis=1).median(); disp=rr.iloc[t-4:t+1].sum().std(ddof=1)
 stress=1.0+np.clip(max(0.,-m5)*8.0 + disp*3.0,0.,2.0)
 f={}
 for s in P:
  r=R[s].iloc[t-59:t+1].dropna()
  if len(r)<50: continue
  vol=P[s].iloc[t-2:t+1].pct_change().abs().mean() # fallback price activity proxy
  base=-float(R[s].iloc[t-2:t+1].sum())
  # stress modulation is common but signal remains cross-sectional
  f[s]=base*stress/(float(r.std(ddof=1))*np.sqrt(3)+1e-9)
 f=pd.Series(f); sig.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index); q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20300307_conditional_stress_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'instruments',len(U),'available',len(px))
