import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
C={}; F={}
for a in assets:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
    d=d[d.date<=cut].set_index('date')
    # close-location reversal: negative prior-day intraday move, scaled by range
    rng=(d.high-d.low).replace(0,np.nan)
    clv=((d.close-d.low)/rng-0.5).clip(-0.5,0.5)
    intraday=d.close/d.open-1
    # positive means buy weak closes / fade intraday move
    F[a]=(-clv * (intraday.abs()/intraday.abs().rolling(20,min_periods=10).median()).clip(0,5))
fac=pd.DataFrame(F).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close
# stress is observable at signal date, robust percentile-like z of VIX level
vz=(vix-vix.rolling(60,min_periods=30).median())/(vix.rolling(60,min_periods=30).std())
stress=(1+vz.clip(lower=0,upper=3)/3).reindex(fac.index)
fac=fac.mul(stress,axis=0)
fac.to_csv('scripts/miner_3_20270325_vix_conditioned_reversal_signal.csv')
prices=pd.DataFrame(C) if C else None
# reload closes for returns
closes={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); closes[a]=d.close
close=pd.DataFrame(closes).reindex(fac.index)
def calc(y):
 vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals,index=dates)
 return s,ns
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h)
 s,ns=calc(y)
 print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('assets',len(assets),'date range',fac.index.min(),fac.index.max())
