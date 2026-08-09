import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# One interpretable idea: 20d momentum weighted by contemporaneous cross-asset breadth.
# Breadth is lagged with the asset signal; forward return starts after signal date.
mom=p.pct_change(20); breadth=(mom>0).sum(axis=1)/mom.notna().sum(axis=1)
factor=mom.mul(2*breadth-1,axis=0).shift(1)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in factor.index:
  x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(vals); print(h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for n in [60,120,250]:
 s=[]
 for dt in factor.index[-n:]:
  z=pd.concat([factor.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('recent',n,'IC',round(np.mean(s),6),'ICIR',round(np.mean(s)/np.std(s,ddof=1),6),'n',len(s))
print('coverage',factor.notna().mean().mean(),'turnover',((factor.rank(axis=1,pct=True).diff().abs()).mean(axis=1)>0.1).mean())
for y,g in pd.Series({dt: v for dt,v in zip(factor.index,[]) }).items(): pass
# annual daily IC
ics=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); print(q.groupby(q.index.year).agg(['mean','count']).round(5).to_string())
