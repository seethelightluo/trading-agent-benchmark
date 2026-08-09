import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
paths=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(x)[:-4] for x in paths]
cl={}; op={}
for p in paths:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date')
 cl[a]=d.close; op[a]=d.open
close=pd.DataFrame(cl).sort_index(); open_=pd.DataFrame(op).reindex(close.index)
r=close.pct_change()
# Contrarian 3-day move, risk scaled. Condition on broad cross-asset dispersion:
# reversal is strongest after synchronized selloffs; suppress in quiet/one-name moves.
breadth=(r.rolling(3).mean()<0).sum(axis=1)/r.notna().sum(axis=1)
stress=(breadth-0.5).abs()+0.5
# smooth broad selloff state, observable at t and applied to all names
sell=(breadth.rolling(5,min_periods=3).mean()).clip(0,1)
state=(0.75+sell).clip(0.75,1.75)
vol=r.rolling(15,min_periods=10).std().replace(0,np.nan)
fac=(-r.rolling(3).sum()/vol)*state.values[:,None]
fac.columns=assets
fac.to_csv('scripts/miner_3_20270325_breadth_conditioned_reversal_signal.csv')
def calc(y):
 vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 return pd.Series(vals,index=dates),ns
for h in [1,5,10]:
 s,ns=calc(close.pct_change(h).shift(-h)); print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(assets),'range',fac.index.min(),fac.index.max())
