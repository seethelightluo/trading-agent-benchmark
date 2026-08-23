import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-10-20')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.close.loc[:cut]
p=pd.DataFrame({s:load(s) for s in U}).sort_index(); r=p.pct_change()
# Acceleration: recent 5-session return minus the average 5-session return over the prior 20 sessions.
# Positive values indicate improving relative trend; median-center cross-sectionally.
recent=p.pct_change(5); prior=(p.shift(5).pct_change(20))/4.0
fac=(recent-prior); fac=fac.sub(fac.median(axis=1),axis=0)
vals_by_h={}
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.DataFrame({'f':fac.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(vals); vals_by_h[h]=(a,ds)
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print(' regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
valid=fac.notna().sum(axis=1); print('coverage',round(valid.mean()/len(U),4),'mean_valid',round(valid.mean(),2),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'period',p.index.min().date(),p.index.max().date())
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('factors/miner_2_20271021_relative_momentum_acceleration_signal.csv',index=False)
