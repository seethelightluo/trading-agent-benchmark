import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# short-term relative momentum acceleration: 5d return minus scaled 20d return, cross-sectional rank-ready
D={s:get_stock_daily_data(s,days=5000) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index()
r=px.pct_change()
# robust acceleration: recent 5d return - 0.25*prior 20d return (avoids duplicate long momentum)
f=r.rolling(5).sum()-0.25*r.shift(5).rolling(20).sum()
rows=[]
for i in range(len(px)-20):
    dt=px.index[i]
    x=f.iloc[i]
    y=r.iloc[i+1:i+11].sum().reindex(U)
    z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    if len(z)>=8: rows.append((dt,len(z),z.x.corr(z.y),z.x.rank().corr(z.y.rank())))
q=pd.DataFrame(rows,columns=['date','n','ic','ric']).dropna()
print('rows',len(px),'dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15)
for h in [5,10,20]:
 rows=[]
 for i in range(len(px)-h):
  x=f.iloc[i]; y=r.iloc[i+1:i+1+h].sum().reindex(U)
  z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append(z.x.corr(z.y))
 a=pd.Series(rows).dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std()*np.sqrt(len(a)),'hit',(a>0).mean())
print('regimes')
for lo,hi in [('2023','2025'),('2026','2028'),('2029','2031'),('2032','2035')]:
 a=q[(q.date>=lo)&(q.date<=hi)].ic; print(lo,hi,len(a),a.mean(),a.mean()/a.std()*np.sqrt(len(a)) if len(a)>1 else np.nan)
# turnover rank signal
rr=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna(); print('turnover',rr.mean())
print('last',px.index[-1])
