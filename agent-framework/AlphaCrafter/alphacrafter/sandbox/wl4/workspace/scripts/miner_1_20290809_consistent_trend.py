import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,4000) for s in A}
c=pd.concat({s:x.set_index('date').close.astype(float) for s,x in D.items() if x is not None},axis=1).sort_index(); r=c.pct_change(fill_method=None)
ret=c.pct_change(60,fill_method=None); cons=r.gt(0).rolling(60,min_periods=45).mean().sub(.5).mul(2); f=(ret*cons).shift(1)
def run(h):
 vals=[]; ns=[]
 for i in range(len(c)-h):
  z=pd.concat([f.iloc[i],c.iloc[i+h]/c.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for n in [250,500]:
  b=a[-n:]; print('recent',n,'IC',round(b.mean(),6),'ICIR',round(b.mean()/b.std(ddof=1),6))
run(5);run(10);run(20)
valid=f.notna().sum(axis=1)>=8; ranks=f.where(valid).rank(axis=1,pct=True)
print('instruments',len(A),'dates',len(c),'coverage',round(f.notna().sum().sum()/f.size,4),'rank_turnover',round(ranks.diff().abs().mean(axis=1).mean(),6),'range',c.index.min(),c.index.max())
