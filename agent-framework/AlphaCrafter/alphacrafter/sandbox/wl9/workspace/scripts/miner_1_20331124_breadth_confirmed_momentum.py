import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2033-11-23')
p={}
for s in U:
 d=pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index(); p[s]=d
p=pd.DataFrame(p).sort_index().loc[:cutoff].ffill(); r20=p.pct_change(20); r60=p.pct_change(60)
breadth=(r20>0).mean(axis=1)
f=(r60/(p.pct_change().rolling(60).std()*np.sqrt(252)+.05)).mul((breadth-.5)*2,axis=0).shift(1)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20331124_breadth_confirmed_momentum_signal.csv',index=False)
print('dates',len(p),'assets',len(U),'range',p.index.min().date(),p.index.max().date())
for h in [10,20,40,60]:
 fr=p.shift(-h).div(p).sub(1); ics=[]; ns=[]; dates=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 a=np.array(ics); ir=np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(len(a))
 print(h,'n_dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(ir,6),'hit',round(np.mean(a>0),4))
 for label,lo,hi in [('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031-32','2031-01-01','2032-12-31'),('2033','2033-01-01','2033-11-23')]:
  q=np.array([v for v,d in zip(a,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]); print(' ',label,len(q),round(np.nanmean(q),6) if len(q) else None)
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
