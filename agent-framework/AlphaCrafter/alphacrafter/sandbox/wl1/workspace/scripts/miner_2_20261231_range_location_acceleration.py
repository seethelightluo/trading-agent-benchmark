import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-12-30'
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change()
# Range-location pressure: recent return relative to distance from rolling high/low, rewarding assets
# rising from lower half with positive short acceleration, while penalizing overextended highs
hi=p.rolling(60,min_periods=30).max(); lo=p.rolling(60,min_periods=30).min()
loc=(p-lo)/(hi-lo+1e-12)
acc=p.pct_change(10)-p.pct_change(30)/3
vol=r.rolling(20,min_periods=10).std()
f=((0.5-loc)*acc/(vol+1e-8)).shift(1)
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h); a=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 x=np.array(a); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.mean(x),6),'ICIR',round(np.mean(x)/np.std(x,ddof=1),6),'hit',round(np.mean(x>0),4))
 for y,g in pd.Series(x,index=ds).groupby(pd.DatetimeIndex(ds).year): print(' year',y,round(g.mean(),6),len(g))
print('coverage',round(f.notna().mean().mean(),6),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
