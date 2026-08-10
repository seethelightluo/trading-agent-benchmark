import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
EQ=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']; DEF=['XAU','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); fwd=px.shift(-1)/px-1
asset20=px.pct_change(20).shift(1); def20=asset20[DEF].mean(axis=1); sig=asset20.sub(def20,axis=0)
breadth=r[EQ].lt(0).sum(axis=1).div(r[EQ].notna().sum(axis=1)).shift(1)
def ic(mask):
 mask=pd.Series(mask,index=sig.index); vals=[]; ns=[]
 for d in sig.index:
  if not bool(mask.loc[d]): continue
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
for name,mask in [('all',pd.Series(True,index=sig.index)),('stress',breadth>=.5),('normal',breadth<.5),('highstress',breadth>=.625)]: print(name,ic(mask))
print('coverage',sig.notna().sum().sum()/(len(sig)*len(U)),'dates',len(sig),'instruments',len(U))
for h in [1,5,10]:
 fh=px.shift(-h)/px-1; vals=[];ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fh.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for y in [2025,2026,2027]: print('year',y,ic(sig.index.year==y))
out=sig.copy();out.index.name='date';out.to_csv('../persistent/factor_signals_miner_1_20270225_defensive_relative_strength.csv')
