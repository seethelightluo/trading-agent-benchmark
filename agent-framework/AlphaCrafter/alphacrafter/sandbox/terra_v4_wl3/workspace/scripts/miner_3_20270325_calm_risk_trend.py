import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(x)[:-4] for x in files]
cl={}; op={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 d=d[d.index<=cut]; cl[a]=d.close; op[a]=d.open
close=pd.DataFrame(cl).sort_index(); ret=close.pct_change()
# One interpretable candidate: medium-term trend continuation, risk-normalized,
# attenuated in elevated VIX (avoid chasing stressed reversals).
vol=ret.rolling(20,min_periods=15).std()
trend=close.pct_change(20)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(close.index)
vz=(vix-vix.rolling(60,min_periods=30).median())/vix.rolling(60,min_periods=30).std()
# weight remains positive, 1 in calm, 0.5 at extreme stress
calm=(1-0.5*(vz.clip(lower=0,upper=3)/3)).fillna(1)
fac=(trend/vol).mul(calm,axis=0)
fac.to_csv('scripts/miner_3_20270325_calm_risk_trend_signal.csv')

def evaluate(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 s=pd.Series(vals,index=ds)
 return s,np.mean(ns)
for h in [1,5,10]:
 s,n=evaluate(h); print('horizon',h,'dates',len(s),'avgN',round(n,2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
s,n=evaluate(1)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('assets',len(assets),'date range',fac.index.min(),fac.index.max())
