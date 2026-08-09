import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
paths=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(x)[:-4] for x in paths]
closes={}; fac={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 d=d[d.index<=cut]
 closes[a]=d.close
 # intermediate-horizon momentum, scaled by realized volatility; use prior completed close
 r=d.close.pct_change()
 mom=d.close.pct_change(20)
 vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
 fac[a]=mom/vol.replace(0,np.nan)
fac=pd.DataFrame(fac).sort_index(); close=pd.DataFrame(closes).reindex(fac.index)
# DXY regime: strengthen trend signal when dollar is calm/trending down, fade when dollar shock is high
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date').close
mr=macro.pct_change()
dxy_vol=mr.rolling(20,min_periods=15).std()
dxy_trend=macro.pct_change(20)
# continuous, bounded regime multiplier, observable at signal date
mult=(1 - 0.75*(dxy_trend/dxy_vol.replace(0,np.nan)).clip(-2,2)/2).clip(0.25,1.75)
fac=fac.mul(mult.reindex(fac.index),axis=0)
fac.to_csv('scripts/miner_1_20270325_dxy_conditioned_momentum_signal.csv')

def calc(y):
 vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals,index=dates); return s,ns
for h in [1,5,10]:
 s,ns=calc(close.pct_change(h).shift(-h))
 print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('assets',len(assets),'date range',fac.index.min(),fac.index.max())
