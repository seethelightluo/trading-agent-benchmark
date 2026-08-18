import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'; END=pd.Timestamp('2034-10-12')
frames={}
for s in U:
 f=os.path.join(P,s+'.csv')
 if os.path.exists(f):
  d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); frames[s]=d.loc[d.date<=END].set_index('date').close.rename(s)
px=pd.concat(frames.values(),axis=1).sort_index(); mom=px.pct_change(20); breadth=mom.gt(0).mean(axis=1)
# No look-ahead: regime and momentum are both lagged before forward return.
factor=mom.mul(np.where(breadth>=0.5,1.0,-1.0),axis=0).shift(1)
ics=[]; cov=[]; turns=[]; dates=[]
for i in range(len(px)-10):
 f=factor.iloc[i]; fw=px.iloc[i+10]/px.iloc[i]-1; z=pd.concat([f,fw],axis=1).dropna()
 if len(z)>=8:
  x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(x): ics.append(x); cov.append(len(z)/15); dates.append(px.index[i])
  if i: turns.append((f.rank(pct=True)-factor.iloc[i-1].rank(pct=True)).abs().mean())
a=np.array(ics)
for n in [120,260,520]:
 q=a[-n:]; print('recent',n,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('dates',len(a),'avg_instruments',np.mean(np.array(cov)*15),'coverage',np.mean(cov),'turnover',np.nanmean(turns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'start',dates[0].date(),'end',dates[-1].date())
pd.DataFrame(factor).to_csv('scripts/artifacts/miner_1_20341012_breadth_regime_momentum_signal.csv'); pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/artifacts/miner_1_20341012_breadth_regime_momentum_ic.csv',index=False)
