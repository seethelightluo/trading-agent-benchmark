import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-05-27')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff();
# intermediate trend: 60d return excluding most recent 10d, scaled by 40d vol, lagged
fac=(np.log(p).diff(60).shift(1)-np.log(p).diff(10).shift(1))/(r.rolling(40).std().shift(1)*np.sqrt(40))
out=[]
for h in [1,3,5,10]:
 vals=[]; ns=[]
 fr=np.log(p).shift(-h)-np.log(p)
 for dt in fac.index:
  a=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a))
 x=np.array(vals); print(h,len(x),round(np.mean(ns),2),round(np.mean(ns)/15,3),round(np.nanmean(x),6),round(np.nanmean(x)/np.nanstd(x,ddof=1),6),round(np.mean(x>0),4))
fac.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330527_intermediate_trend_signal.csv',index=False)
