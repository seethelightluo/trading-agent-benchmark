import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change(); E=U[:8]
base=p.pct_change(20).sub(p.pct_change(20)[E].mean(axis=1),axis=0)
down=r.where(r<0).rolling(20,min_periods=10).std()
f=(base/(down+1e-5)).shift(1)
fr=p.pct_change(10).shift(-10); allics=[]; ns=[]; ds=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  allics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(d)
a=np.array(allics); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0)); print('coverage',f.notna().mean().mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for y,g in pd.Series(a,index=ds).groupby(pd.DatetimeIndex(ds).year): print(y,g.mean(),len(g))
