import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change();
# low downside-risk: inverse downside deviation, blended with stable total vol; lagged
for n in [10,20,40]:
 dn=r.where(r<0).rolling(n).std(); tv=r.rolling(n).std(); f=(1/(.7*dn+.3*tv)).shift(1)
 print('WINDOW',n)
 for h in [1,3,5,10]:
  vals=[]; ns=[]
  for i in range(len(P)-h):
   z=pd.concat([f.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  q=pd.Series(vals).dropna(); print(h,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),3))
