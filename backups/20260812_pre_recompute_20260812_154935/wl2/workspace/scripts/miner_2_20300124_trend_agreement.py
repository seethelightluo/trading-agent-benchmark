import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# Risk-adjusted trend agreement: 20-day momentum scaled by downside volatility,
# gated by agreement of 5-, 20-, and 60-day return signs. All windows are lagged
# at the decision date and forward returns begin on the next completed day.
for t in range(65,len(P)-11):
 v={}
 for s in P:
  r5=R[s].iloc[t-4:t+1].dropna(); r20=R[s].iloc[t-19:t+1].dropna(); r60=R[s].iloc[t-59:t+1].dropna()
  if len(r5)<4 or len(r20)<17 or len(r60)<50: continue
  dn=r20[r20<0]
  dv=dn.std() if len(dn)>=3 else r20.std()
  if not np.isfinite(dv) or dv<=1e-8: continue
  m20=r20.sum()/dv
  signs=np.sign([r5.sum(),r20.sum(),r60.sum()])
  agree=(np.sum(signs==np.sign(r20.sum()))/3.0)
  v[s]=m20*agree
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
S=pd.DataFrame(sig); S.to_csv('scripts/miner_2_20300124_trend_agreement_signal.csv',index_label='date')
print('signal_rows',len(S),'turnover',np.nanmean((S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)).values),'instruments',len(U),'available',len(px))
