import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); xs=r.sub(r.mean(axis=1),axis=0)
# Dispersion-conditioned short reversal: only activate after unusually dispersed 20d cross-section.
disp=xs.std(axis=1); gate=(disp>disp.rolling(120,min_periods=80).quantile(.6)).shift(1)
vol=xs.rolling(20,min_periods=15).std(); f=(-xs.rolling(5,min_periods=5).sum()/vol).shift(1).mul(gate.astype(float),axis=0)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and f.loc[dt].abs().sum()>0: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics).dropna(); print('assets',len(raw),'calendar_dates',len(px),'active_valid_dates',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15)); print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns),'active',np.mean(f.abs().sum(axis=1)>0))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b and f.loc[d].abs().sum()>0:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331223_dispersion_reversal_signal.csv',index=False)
