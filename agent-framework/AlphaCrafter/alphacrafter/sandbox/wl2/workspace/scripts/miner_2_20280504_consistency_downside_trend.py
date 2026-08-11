import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=2600)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=2600)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill(); r=P.pct_change()
# Novel persistence/anti-whipsaw signal: 60d return weighted by fraction of positive sessions,
# then made cross-sectional relative and scaled by downside deviation.
ret60=P.pct_change(60); consistency=(r.gt(0).rolling(60,min_periods=40).mean()-0.5)*2
raw=ret60*consistency
F=raw.sub(raw.median(axis=1),axis=0)/(r.where(r<0,0).rolling(60,min_periods=40).std()+1e-6)
for h in [1,3,5,10]:
 y=P.pct_change(h).shift(-h+1); a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=z.f.clip(z.f.quantile(.05),z.f.quantile(.95)).corr(z.y)
   if pd.notna(q): a.append(q); ns.append(len(z))
 x=pd.Series(a); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'std',round(x.std(ddof=1),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'coverage',round(F.notna().mean().mean(),6),'avgN',round(F.notna().sum(axis=1).mean(),2),'instruments',len(D),'period',P.index.min(),P.index.max())
for lo,hi in [(2020,2022),(2023,2025),(2026,2028)]:
 a=[]
 for i in range(len(P)-1):
  if lo<=P.index[i].year<=hi:
   z=pd.concat([F.iloc[i],P.pct_change(1).shift(-0).iloc[i]],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1]); a += [q] if pd.notna(q) else []
 print('regime',lo,'-',hi,'dates',len(a),'IC',round(np.mean(a),6) if a else None,'hit',round(np.mean(np.array(a)>0),4) if a else None)
