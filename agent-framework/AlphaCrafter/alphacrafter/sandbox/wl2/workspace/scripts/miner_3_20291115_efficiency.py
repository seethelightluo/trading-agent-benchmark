import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Directional efficiency: net 20d move divided by path length, scaled by inverse volatility.
# It rewards persistent trends rather than noisy returns.
net=P.pct_change(20); path=R.abs().rolling(20,min_periods=15).sum(); vol=R.rolling(20,min_periods=15).std()
F=(net/(path+1e-9))/(vol+1e-9)
rows=[]; sig=[]
for t in range(30,len(P)-11):
 v=F.iloc[t].replace([np.inf,-np.inf],np.nan).dropna()
 for h in (1,5,10):
  q=pd.concat([v,R.iloc[t+1:t+h+1].sum().reindex(v.index)],axis=1).dropna()
  if len(q)>=8: rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 sig.append(v.rename(P.index[t]))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in (1,5,10):
 a=o[o.h==h].set_index('date').ic
 print('h',h,'dates',len(a),'avgN',round(o[o.h==h].n.mean(),2),'coverage',round(o[o.h==h].n.mean()/len(U),4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
 for c in ['2026-01-01','2028-01-01','2029-01-01','2029-07-01']:
  b=a[a.index>=c]; print(c,len(b),round(b.mean(),5),round(b.mean()/b.std(ddof=1),5) if len(b)>1 else None)
S=pd.DataFrame(sig); S.to_csv('scripts/miner_3_20291115_efficiency_signal.csv',index_label='date')
print('turnover',np.nanmean(S.diff().abs().sum(axis=1)/(S.abs().sum(axis=1)+1e-9)),'assets',len(P.columns),'dates',len(P),'signal_rows',len(S))
