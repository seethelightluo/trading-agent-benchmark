import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(x)[:-4] for x in files]
closes={}; rets={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 closes[a]=d.close; rets[a]=d.close.pct_change()
close=pd.DataFrame(closes).sort_index(); r=pd.DataFrame(rets).reindex(close.index)
# Fade 3-day excess return; amplify only when cross-asset daily dispersion is unusually high.
r3=close.pct_change(3); excess=r3.sub(r3.median(axis=1),axis=0)
disp=r.rolling(20,min_periods=10).std().mean(axis=1)
dz=(disp-disp.rolling(60,min_periods=30).median())/(disp.rolling(60,min_periods=30).std())
amp=(1+dz.clip(lower=0,upper=3)/3)
fac=-excess.div(close.pct_change().rolling(20,min_periods=10).std()).mul(amp,axis=0)
fac.to_csv('scripts/miner_3_20270325_dispersion_conditioned_reversal_signal.csv')
def calc(y):
 vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 return pd.Series(vals,index=dates),ns
for h in [1,5,10]:
 s,ns=calc(close.pct_change(h).shift(-h)); print('horizon',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),7),'ICIR',round(s.mean()/s.std(ddof=1),7),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'dates',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7))
print('coverage',round(fac.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'assets',len(assets),'range',fac.index.min(),fac.index.max())
