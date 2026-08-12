import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change(); r20=prices.pct_change(20); vol20=ret.rolling(20).std()*np.sqrt(252)
score=r20.div(vol20.replace(0,np.nan)); medv=vol20.median(axis=1)
compression=medv.rolling(5).median().div(medv.rolling(60).median())<0.85
breadth=r20.gt(0).mean(axis=1)>=0.50; active=compression & breadth
f=score.where(active, np.nan); fwd=prices.shift(-1).div(prices)-1
ics=[]; dates=[]; ns=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); ns.append(len(z))
ics=np.array(ics); dates=pd.DatetimeIndex(dates)
def stats(a): return (round(a.mean(),6),round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),len(a)) if len(a) else ('nan','nan',0)
print('candidate=vol_compression_gated_risk_adjusted_20d_momentum')
print('dates',len(ics),'instruments_avg',round(np.mean(ns),3),'coverage',round(f.notna().sum().sum()/f.size,4),'active_dates',round(active.mean(),4))
print('IC ICIR n',stats(ics),'hit',round(np.mean(ics>0),4))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2029-12-31'),('2030','2032-04-01')]:
 m=(dates>=pd.Timestamp(lo))&(dates<=pd.Timestamp(hi)); print(lo,hi,'IC ICIR n',stats(ics[m]))
for h in [3,5,10]:
 yy=prices.shift(-h).div(prices)-1; aa=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC n',stats(np.array(aa)))
out=f.copy(); out.insert(0,'date',out.index); out.to_csv('scripts/miner_1_20320401_volcompression_momentum_signal.csv',index=False)
