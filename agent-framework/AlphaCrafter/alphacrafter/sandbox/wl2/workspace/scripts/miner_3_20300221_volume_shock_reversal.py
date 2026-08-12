import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for s in U:
 d=get_stock_daily_data(s,2400)
 if d is None or len(d)<100: d=get_index_daily_data(s,2400)
 if d is not None:
  d=d.set_index('date'); px[s]=d.close.astype(float); vol[s]=pd.to_numeric(d.volume,errors='coerce')
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index); R=P.pct_change()
rows=[]; signals=[]
# Volume-confirmed short shock reversal: fade the 3-session return, with
# volume surprise as a bounded confidence weight. All windows end at t.
for t in range(65,len(P)-11):
 f={}
 for s in P:
  r=R[s].iloc[t-2:t+1]; vv=V[s].iloc[t-19:t+1]
  if r.notna().sum()<3 or vv.notna().sum()<15: continue
  base=vv.iloc[:-3].median()
  if not np.isfinite(base) or base<=0: continue
  shock=float(r.sum()); vs=float(vv.iloc[-3:].mean()/base-1)
  # fade shocks, emphasize unusual volume, but bound the amplification
  f[s]=-shock*(1+np.clip(vs,-0.5,2.0))
 f=pd.Series(f)
 signals.append(f.rename(P.index[t]))
 for h in (1,5,10):
  fw=R.iloc[t+1:t+h+1].sum().reindex(f.index)
  q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(signals); S.to_csv('scripts/miner_3_20300221_volume_shock_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'instruments',len(U),'available',len(px),'volume_nonnull',round(V.notna().mean().mean(),4))
