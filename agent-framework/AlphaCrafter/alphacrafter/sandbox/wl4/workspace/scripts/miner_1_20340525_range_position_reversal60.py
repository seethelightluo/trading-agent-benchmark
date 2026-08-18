import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; ds={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d['date']=pd.to_datetime(d.date);ds[s]=d.set_index('date').close.sort_index()
pd0=pd.DataFrame(ds).sort_index().loc[:'2034-05-25']; hi=pd0.rolling(60,min_periods=60).max();lo=pd0.rolling(60,min_periods=60).min(); f=(-(pd0-lo)/(hi-lo).replace(0,np.nan)).shift(1); fr=pd0.shift(-10)/pd0-1
out=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=np.array([x[1] for x in out]);print('dates',len(a),'avgN',np.mean([x[2] for x in out]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',len(a)/len(pd0))
for n in [120,260,520,780]:
 q=a[-n:];print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
os.makedirs('scripts/artifacts',exist_ok=True);pd.DataFrame(out,columns=['date','ic','n']).to_csv('scripts/artifacts/miner_1_20340525_range_position_reversal60_ic.csv',index=False);f.to_csv('scripts/artifacts/miner_1_20340525_range_position_reversal60_signal.csv')
