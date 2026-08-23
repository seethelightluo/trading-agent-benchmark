import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-25'); px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[a]=d['close'].loc[:cut]
P=pd.concat(px,axis=1).sort_index(); R=P.pct_change()
med=R.median(axis=1); resid=R.sub(med,axis=0)
vol=R.rolling(20,min_periods=12).std().shift(1)
f=-(resid.rolling(3,min_periods=3).sum().shift(1))/vol
for h in [1,5,10]:
 ic=[]; counts=[]; fr=P.pct_change(h).shift(-h)
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): ic.append(v); counts.append(len(z))
 arr=np.array(ic); print('h',h,'dates',len(arr),'avg_names',round(np.mean(counts),2),'coverage',round(np.mean(counts)/15,4),'IC',round(np.mean(arr),6),'ICIR',round(np.mean(arr)/(np.std(arr,ddof=1)+1e-12),6),'hit',round(np.mean(arr>0),4))
r=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(np.mean(np.abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover',round(float(np.mean(turn)),6),'period',P.index.min().date(),P.index.max().date())
