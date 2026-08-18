import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-05-12'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); resid=r.sub(r.mean(axis=1),axis=0); disp=resid.std(axis=1)
# medium-horizon residual reversal, volatility-scaled and restricted to unusually dispersed sessions
x=resid.rolling(15,min_periods=10).sum(); rv=resid.rolling(60,min_periods=30).std(); active=(disp>disp.rolling(120,min_periods=60).quantile(.65)).shift(1); f=(-x/rv).mul(active,axis=0).shift(1)
ics=[];ns=[];dates=[];turn=[]; fr=np.log(px.shift(-10)/px)
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z));dates.append(d)
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append((z.iloc[:,0].rank()-z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics,index=pd.to_datetime(dates)).dropna();print('assets',len(raw),'dates',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4),'turn',round(np.mean(turn),4))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))];print(a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
for h in [5,20]:
 frh=np.log(px.shift(-h)/px);v=[]
 for d in f.index:
  z=pd.concat([f.loc[d],frh.loc[d]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(pd.Series(v).mean(),6),len(v))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20330513_medium_dispersion_reversal_signal.csv',index=False)
print('signal_path=scripts/miner_2_20330513_medium_dispersion_reversal_signal.csv')
