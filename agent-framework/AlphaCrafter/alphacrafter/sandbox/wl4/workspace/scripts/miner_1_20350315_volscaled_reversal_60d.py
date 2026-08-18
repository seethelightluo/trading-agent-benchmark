import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';px={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change(); f=(-(P.pct_change(60)/(R.rolling(20).std()*np.sqrt(20)))).shift(1); fr=P.shift(-10)/P-1
ics=[];dates=[];ns=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):ics.append(q);dates.append(dt);ns.append(len(z))
a=np.array(ics);print('dates',len(a),'avg_inst',np.mean(ns),'coverage',np.isfinite(f.loc[dates]).sum().sum()/(len(dates)*len(U)));print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
rr=f.rank(axis=1,pct=True);tv=[]
for d1,d2 in zip(dates[:-1],dates[1:]):
 x,y=rr.loc[d1],rr.loc[d2];ok=x.notna()&y.notna();tv.append(np.abs(x[ok]-y[ok]).mean())
print('turnover',np.mean(tv))
for n in [120,260,520,780]:
 b=a[-n:];print('recent',n,'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1))
os.makedirs('scripts/artifacts',exist_ok=True);pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/artifacts/miner_1_20350315_volscaled_reversal_60d_ic.csv',index=False);pd.DataFrame({'date':dates}).to_csv('scripts/artifacts/miner_1_20350315_volscaled_reversal_60d_signal.csv',index=False)
