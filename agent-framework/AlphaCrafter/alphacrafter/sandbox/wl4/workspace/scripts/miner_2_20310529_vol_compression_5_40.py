import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:D[s]=get_index_daily_data(s,days=4000)
 except Exception:D[s]=get_stock_daily_data(s,days=4000)
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=px.pct_change()
# Favor assets whose recent 5d volatility is compressed versus their 40d baseline; lagged one session.
f=-(r.rolling(5,min_periods=4).std()/(r.rolling(40,min_periods=25).std()+1e-12)).shift(1)
def calc(H,lo=0):
 fw=px.shift(-H)/px-1; vals=[]; ns=[]
 for i in range(lo,len(px)-H):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); ns.append(len(z))
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns)
print('cutoff',px.index.max().date(),'dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:print('H%d'%h,calc(h))
for w in [365,730,1095]:print('recent%d'%w,calc(10,max(0,len(px)-w-10)))
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'rank_turnover',rank.diff().abs().mean(axis=1).mean())