import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x)>100:return x[['date','close']]
  except Exception: pass
parts=[]
for s in U:
 x=load(s)
 if x is not None:
  x=x.set_index('date').sort_index(); r=x.close.pct_change()
  parts.append(pd.DataFrame({s:r,s+'_r':r,s+'_v':r.rolling(20).std(),s+'_r5':x.close.pct_change(5)}))
d=pd.concat(parts,axis=1).sort_index()
R=d[[s+'_r' for s in U]].rename(columns=lambda x:x[:-2])
# residualized 5d reversal, scaled by trailing vol; activate when cross-asset dispersion is elevated
raw=d[[s+'_r5' for s in U]].rename(columns=lambda x:x[:-3])
vol=d[[s+'_v' for s in U]].rename(columns=lambda x:x[:-2])
res=raw.sub(raw.mean(axis=1),axis=0)
disp=R.std(axis=1); gate=(disp/disp.rolling(60).median()).clip(0.5,2.0)
f=(-res/(vol*np.sqrt(5)+1e-8)).mul(gate,axis=0).rolling(2,min_periods=2).mean().shift(1)
# forward 1d admission horizon and longer decay
out=[]
for h in [1,5,10,20]:
 fr=(1+R).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=pd.Series(vals); recent=q.tail(250)
 print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'min_n',min(ns),'IC %.6f ICIR %.6f hit %.4f recent %.6f/%.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),recent.mean(),recent.mean()/recent.std(ddof=1)))
 if h==1:
  print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
# dispersion regime for 1d
fr=R.shift(-1); q=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(q,columns=['date','ic']).set_index('date'); med=disp.reindex(a.index).median()
for n,m in [('low',disp.reindex(a.index)<=med),('high',disp.reindex(a.index)>med)]:
 x=a.ic[m].dropna();print(n,'dates',len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
