import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
trend=p.pct_change(40); comp=(v60/(v20+1e-12)).clip(.5,2.0)
# Breadth-conditioned compression momentum: use only lagged cross-sectional breadth;
# damp trend during broad bearish regimes and emphasize it during broad positive regimes.
breadth=(trend>0).mean(axis=1)
gate=(0.65+0.70*breadth).clip(.65,1.35)
raw=trend*comp.mul(gate,axis=0)
f=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1),axis=0)
fr=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in rows]); dates=[x[0] for x in rows]
print('factor breadth_conditioned_compression_momentum')
print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'start',min(dates),'end',max(dates))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([x[2] for x in rows])/15)
for n in [365,750,1260]:
 q=a[-n:]; print('recent',n,q.mean(),q.mean()/q.std(ddof=1),len(q))
for h in [1,5,20]:
 yy=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(q),len(q))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20340413_breadth_conditioned_compression_signal.csv')
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_2_20340413_breadth_conditioned_compression_ic.csv',index=False)
