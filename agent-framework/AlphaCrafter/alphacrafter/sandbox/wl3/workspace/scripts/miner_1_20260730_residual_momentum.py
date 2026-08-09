import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root=Path('../persistent/stock_data')
def load(s): return pd.read_csv(root/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2026-07-15']
p=pd.concat([load(s).rename(s) for s in syms],axis=1); r=p.pct_change(fill_method=None); m=r.mean(axis=1); mv=m.rolling(60,min_periods=45).var()
beta=pd.DataFrame({s:r[s].rolling(60,min_periods=45).cov(m).div(mv) for s in syms}); fac=p.pct_change(20,fill_method=None)-beta.mul(m.rolling(20).sum(),axis=0); fwd=r.shift(-1)
def ev(y,idx=None):
 if idx is None: idx=fac.index
 q=[];ns=[]
 for dt in idx:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q);return len(q),round(np.mean(ns),2),round(np.mean(ns)/15,4),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5),round(np.mean(q>0),4)
print('daily',ev(fwd));print('5d',ev(p.pct_change(5,fill_method=None).shift(-5)));print('10d',ev(p.pct_change(10,fill_method=None).shift(-10)))
print('turn',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5),'corr20',round(fac.stack().corr(p.pct_change(20).stack()),4),'corrrev',round(fac.stack().corr((-p.pct_change(5)).stack()),4))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:print(a,b,ev(fwd,fac.loc[a:b].index))
