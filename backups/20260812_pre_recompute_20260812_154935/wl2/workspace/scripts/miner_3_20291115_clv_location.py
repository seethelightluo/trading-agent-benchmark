import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: D[s]=d.set_index('date')
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); R=P.pct_change()
# One interpretable idea: persistent close-location value (CLV), smoothed over 10d
# and volatility normalized. High CLV means closes near daily highs; low means near lows.
clv={}
for s,d in D.items():
 den=(d.high-d.low).replace(0,np.nan)
 clv[s]=((2*d.close-d.high-d.low)/den).rolling(10,min_periods=7).mean()
C=pd.DataFrame(clv).reindex(P.index); V=R.rolling(20,min_periods=12).std()
rows=[]; sig=[]
for t in range(30,len(P)-11):
 v=(C.iloc[t]/(V.iloc[t]+1e-9)).replace([np.inf,-np.inf],np.nan).dropna()
 for h in (1,5,10):
  f=R.iloc[t+1:t+h+1].sum().reindex(v.index)
  q=pd.concat([v,f],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(v.rename(P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(o[o.h==h].n.mean(),2),'coverage',round(o[o.h==h].n.mean()/len(U),4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
 for c in ['2026-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),5),round(b.mean()/b.std(ddof=1),5) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20291115_clv_location_signal.csv',index_label='date')
print('turnover',np.nanmean(S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)))
print('assets',len(P.columns),'dates',len(P),'signal_rows',len(S))
