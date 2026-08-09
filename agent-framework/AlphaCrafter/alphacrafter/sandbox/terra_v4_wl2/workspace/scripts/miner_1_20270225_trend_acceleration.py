import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); f=(r.rolling(5).sum()-r.rolling(20).sum()/4).shift(1)
# cross-sectional demean; forward returns at 1,3,5 sessions
for h in [1,3,5]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]; dates=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(d)
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(f.notna().sum().sum()/(len(U)*len(f)),4),'turnover',round(np.mean((f.rank(axis=1,pct=True).diff().abs()).sum(axis=1).dropna()/len(U)),4),'period',px.index.min(),px.index.max())
