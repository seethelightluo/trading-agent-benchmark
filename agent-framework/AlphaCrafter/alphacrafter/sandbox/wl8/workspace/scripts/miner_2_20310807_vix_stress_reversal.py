import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl=pd.DataFrame({s:get_stock_daily_data(s,3000).set_index('date')['close'].astype(float) for s in U}).sort_index()
v=get_index_daily_data('VIX',3000).set_index('date')['close'].astype(float).sort_index()
r=cl.pct_change(10); vol=cl.pct_change().rolling(20,min_periods=10).std()*np.sqrt(252)
level=v.rolling(252,min_periods=60).rank(pct=True).reindex(cl.index)
# Contrarian signal only in objectively stressed VIX regime; lag one completed session.
g=((level-.75)/.25).clip(0,1)
sig=(-r/vol.replace(0,np.nan)).mul(g,axis=0).shift(1)
ics=[]; ns=[]; tos=[]; prev=None
for d in cl.index:
 z=pd.concat([sig.loc[d],(cl.shift(-10)/cl-1).loc[d]],axis=1).dropna()
 if len(z)>=8 and sig.loc[d].abs().sum()>0:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append((d,c));ns.append(len(z))
 rank=sig.loc[d].rank(pct=True)
 if prev is not None:
  q=pd.concat([rank,prev],axis=1).dropna()
  if len(q):tos.append(float((q.iloc[:,0]-q.iloc[:,1]).abs().mean()))
 prev=rank
D=np.array([d for d,c in ics]); X=np.array([c for d,c in ics])
def f(a): return (float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0)),len(a)) if len(a)>1 else (np.nan,)*3+(len(a),)
print('dates',len(X),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'turnover',np.mean(tos))
print('10d',f(X))
for n in [180,360,720]: print('recent',n,f(X[D>=D[-1]-pd.tseries.offsets.BDay(n)]))
for h in [5,20]:
 y=[]
 for d in cl.index:
  z=pd.concat([sig.loc[d],(cl.shift(-h)/cl-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and sig.loc[d].abs().sum()>0: y.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,f(np.array(y)))
sig.index.name='date'; sig.to_csv('scripts/miner_2_20310807_vix_stress_reversal_signal.csv')
icsdf=pd.DataFrame({'date':D,'ic':X}); icsdf.to_csv('scripts/miner_2_20310807_vix_stress_reversal_ic.csv',index=False)
