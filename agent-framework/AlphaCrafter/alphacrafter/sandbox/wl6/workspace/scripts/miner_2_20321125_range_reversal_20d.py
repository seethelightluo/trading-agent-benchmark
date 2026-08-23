import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2032-11-24']; r=p.pct_change()
ret20=p.pct_change(20); hi60=p.rolling(60,min_periods=40).max(); lo60=p.rolling(60,min_periods=40).min(); rng=(hi60-lo60).replace(0,np.nan)
# Reversal is strongest when price is near a range extreme, scaled by range width
loc=(p-lo60)/rng
sig=-ret20*(0.5+abs(loc-0.5)*2)
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; ic=[];ns=[];ds=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(p.index[i])
 x=np.array(ic);print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4));print('regimes',pd.Series(x,index=pd.DatetimeIndex(ds)).groupby(lambda z:z.year).mean().round(5).to_dict())
u=[]
for i in range(1,len(sig)):
 z=pd.concat([sig.iloc[i-1].rank(pct=True),sig.iloc[i].rank(pct=True)],axis=1).dropna()
 if len(z):u.append(np.mean(abs(z.iloc[:,1]-z.iloc[:,0])))
print('turnover',round(float(np.mean(u)),6),'data_dates',len(p),'universe',len(A))
