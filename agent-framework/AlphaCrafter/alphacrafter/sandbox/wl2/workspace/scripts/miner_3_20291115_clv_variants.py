import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:D[s]=d.set_index('date')
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index();R=P.pct_change()
# Smoothed close-location value, a price-action persistence measure.
for w in (5,10,15,20):
 C=pd.DataFrame({s:((2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan)).rolling(w,min_periods=max(4,w//2)).mean() for s,d in D.items()}).reindex(P.index)
 rows=[]
 for t in range(30,len(P)-11):
  v=C.iloc[t].dropna()
  for h in (5,10):
   q=pd.concat([v,R.iloc[t+1:t+h+1].sum().reindex(v.index)],axis=1).dropna()
   if len(q)>=8:rows.append((P.index[t],h,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
 o=pd.DataFrame(rows,columns=['date','h','n','ic'])
 for h in (5,10):
  a=o[o.h==h].ic;print('w',w,'h',h,'dates',len(a),'N',round(o[o.h==h].n.mean(),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
