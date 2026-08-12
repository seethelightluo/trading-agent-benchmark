import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# Downside-asymmetry recovery: favor assets whose recent drawdown is recovering,
# but penalize names whose downside tail is large relative to total volatility.
# All inputs end at t; forward returns begin at t+1.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  r10=R[s].iloc[t-9:t+1].dropna(); r30=R[s].iloc[t-29:t+1].dropna(); r60=R[s].iloc[t-59:t+1].dropna()
  if len(r10)<9 or len(r30)<27 or len(r60)<50: continue
  close=P[s].iloc[:t+1]
  peak=close.iloc[-31:].max(); dd=close.iloc[-1]/peak-1
  # recovery is positive recent return after a local drawdown; asymmetry is downside tail share
  rec=r10.sum()/max(r30.std(),1e-8)
  dn=r30[r30<0]
  downvar=(dn**2).mean() if len(dn) else r30.var()
  tail=np.sqrt(downvar)/max(r30.std(),1e-8)
  v[s]=rec*(1+dd)* (2-tail)
 for h in (1,5,10):
  q=pd.concat([pd.Series(v),R.iloc[t+1:t+h+1].sum().reindex(v)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20300110_downside_asymmetry_recovery_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values))
print('instruments',len(U),'available',len(px))
