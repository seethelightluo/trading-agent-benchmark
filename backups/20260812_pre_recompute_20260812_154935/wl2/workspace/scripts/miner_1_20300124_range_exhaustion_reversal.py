import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date')
P=pd.DataFrame({s:d.close.astype(float) for s,d in px.items()}).sort_index(); R=P.pct_change()
rows=[]; sig=[]
# Range-normalized exhaustion reversal: fade a recent directional move when
# it consumed an unusually large share of its recent true range, with a mild
# close-location confirmation. Every input ends on t.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  r=R[s]; d=px[s]
  r5=r.iloc[t-4:t+1].dropna(); r20=r.iloc[t-19:t+1].dropna()
  if len(r5)<4 or len(r20)<18: continue
  vol=r20.std()
  if vol<=1e-8: continue
  move=r5.sum()/vol
  hi=d.high.iloc[t-19:t+1].astype(float); lo=d.low.iloc[t-19:t+1].astype(float); cl=d.close.iloc[t-19:t+1].astype(float)
  rng=(hi-lo).replace(0,np.nan); clv=((2*cl-hi-lo)/rng).clip(-1,1)
  # exhaustion is directional move times recent candle pressure; reverse it
  pressure=clv.iloc[-5:].mean()
  v[s]=-move*(0.6+0.4*abs(pressure))
 for h in (1,5,10):
  f=pd.Series(v); fw=R.iloc[t+1:t+h+1].sum().reindex(f.index)
  q=pd.concat([f,fw],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(pd.Series(v,name=P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 z=o[o.h==h]; a=z.set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/len(U),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for c in ['2025-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),6),round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_1_20300124_range_exhaustion_reversal_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values),'instruments',len(U),'available',len(px))
