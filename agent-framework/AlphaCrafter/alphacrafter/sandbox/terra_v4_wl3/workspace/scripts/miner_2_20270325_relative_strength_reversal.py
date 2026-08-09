import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
cl={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); cl[a]=d[d.index<=cut].close
close=pd.DataFrame(cl).sort_index(); r5=close.pct_change(5); med=r5.median(axis=1); vol=close.pct_change().rolling(20,min_periods=15).std(); fac=(-(r5.sub(med,axis=0))/vol.clip(lower=.002)).replace([np.inf,-np.inf],np.nan)
fac.to_csv('scripts/miner_2_20270325_relative_strength_reversal_signal.csv')
def calc(y):
 vals=[];ns=[];dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 return pd.Series(vals,index=dates),ns
for h in [1,5,10]:
 s,ns=calc(close.pct_change(h).shift(-h)); print('horizon',h,'dates',len(s),'range',s.index.min(),s.index.max(),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo[:4],len(q),round(q.mean(),7) if len(q) else None)
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(assets))
