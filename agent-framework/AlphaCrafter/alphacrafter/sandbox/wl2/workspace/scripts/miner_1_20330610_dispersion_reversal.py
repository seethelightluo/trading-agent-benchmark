import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-10')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff(); cs=r.sub(r.mean(axis=1),axis=0); disp=cs.std(axis=1).rolling(20,min_periods=15).mean().shift(1); threshold=disp.rolling(252,min_periods=100).quantile(.75)
base=-cs.rolling(3,min_periods=3).sum().shift(1); fac=base.where(disp>threshold)
fr=np.log(p).shift(-3)-np.log(p); vals=[];ns=[]; active=0
for dt in fac.index:
 a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a)); active+=1
x=np.asarray(vals); print('dates',len(x),'avgN',np.mean(ns),'coverage_active',np.mean(ns)/15,'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20330610_dispersion_reversal_signal.csv',index=False)
