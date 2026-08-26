import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
p=pd.DataFrame({s:d.set_index('date').close for s,d in D.items()}).sort_index(); r=p.pct_change()
# idiosyncratic reversal: remove 60d rolling equal-weight benchmark return, then reverse recent 10d and volatility-normalize
b=r.mean(axis=1); res=r.sub(b,axis=0); sig=res.rolling(20).std(); f=-res.rolling(10).sum()/sig
for h in [5,10,20]:
 a=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('x'),r.iloc[i+1:i+1+h].sum().rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.x.corr(z.y));ns.append(len(z))
 a=pd.Series(a).dropna();print(h,len(a),np.mean(ns),np.mean(a),np.mean(a)/a.std()*np.sqrt(len(a)),(a>0).mean())
q=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('x'),r.iloc[i+1:i+11].sum().rename('y')],axis=1).dropna()
 if len(z)>=8:q.append((p.index[i],z.x.corr(z.y)))
q=pd.DataFrame(q,columns=['date','ic'])
for lo,hi in [('2026','2028'),('2029','2031'),('2032','2035')]:
 a=q[(q.date>=lo)&(q.date<=hi)].ic;print(lo,len(a),a.mean())
print('coverage',f.notna().mean(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
