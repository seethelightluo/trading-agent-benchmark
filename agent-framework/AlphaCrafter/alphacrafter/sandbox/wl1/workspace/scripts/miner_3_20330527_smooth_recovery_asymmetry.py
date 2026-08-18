import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-05-26'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date);raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
# Smooth unconditional residual recovery: medium pullback, confirmed by short rebound, risk-adjusted by downside asymmetry.
down=(-resid.where(resid<0)).rolling(60,min_periods=30).mean(); up=resid.where(resid>0).rolling(60,min_periods=30).mean(); asym=(down/(up+1e-8)).clip(.5,3)
rv=resid.rolling(40,min_periods=25).std(); pull=(-resid.rolling(20,min_periods=15).sum()/((rv*np.sqrt(20))+1e-8)); confirm=resid.rolling(5,min_periods=4).sum()/(rv*np.sqrt(5)+1e-8)
f=(pull+0.35*confirm)/(1+0.25*asym); f=f.shift(1); fr=np.log(px.shift(-10)/px)
vals=[]; ns=[]; turns=[]; dates=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals,index=pd.to_datetime(dates)).dropna();print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4),'turn',round(np.mean(turns),4))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))];print(a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20330527_smooth_recovery_asymmetry_signal.csv',index=False)
