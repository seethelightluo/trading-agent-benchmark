import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2033-03-02']; ret=p.pct_change()
mom=p.pct_change(20); vol=ret.rolling(30).std()*np.sqrt(252)
# Continuous breadth state: cross-sectional breadth is centered at 50%,
# clipped to [0.5,1.5], and smoothly scales risk-adjusted momentum.
breadth=(mom>0).sum(axis=1)/mom.notna().sum(axis=1)
scale=(0.5+2*(breadth-0.5)).clip(0.5,1.5)
sig=mom/vol.mul(np.sqrt(20),axis=0)
sig=sig.mul(scale,axis=0)
allvals={}
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(p.index[i])
 x=np.asarray(vals); allvals[h]=x
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4))
 print('regimes',pd.Series(x,index=pd.DatetimeIndex(dates)).groupby(lambda z:z.year).mean().round(5).to_dict())
u=[]
for i in range(1,len(sig)):
 z=pd.concat([sig.iloc[i-1].rank(pct=True),sig.iloc[i].rank(pct=True)],axis=1).dropna()
 if len(z): u.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
print('turnover',round(float(np.mean(u)),6),'instruments',len(A),'last_date',p.index[-1].date())
