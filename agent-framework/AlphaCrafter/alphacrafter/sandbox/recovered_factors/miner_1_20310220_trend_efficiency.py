import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in syms}
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); dates=p.index
# Trend efficiency: directional 15d move divided by path length; rewards persistent trends,
# then remove equal-weight market component to isolate cross-asset relative behavior.
path=r.abs().rolling(15).sum(); raw=p.pct_change(15)/path
mkt=raw.mean(axis=1); f=raw.sub(mkt,axis=0)
# evaluate all horizons, date/regime stability
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'meanIC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'meanN',np.mean(ns))
# regimes H10/H20
h=10; fr=p.shift(-h)/p-1; out=[]
for i in range(len(p)-h):
 z=pd.concat([f.iloc[i],fr.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:out.append((dates[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
v=pd.DataFrame(out,columns=['date','ic']).set_index('date')
for a,b in [('2020','2023'),('2024','2027'),('2028','2031'),('2030-08','2031-01'),('2030-10','2031-02')]:
 q=v.loc[a:b,'ic']; print('REG',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'meanN',f.notna().sum(axis=1).mean(),'turn10',f.rank(axis=1).diff(10).abs().mean().mean())
