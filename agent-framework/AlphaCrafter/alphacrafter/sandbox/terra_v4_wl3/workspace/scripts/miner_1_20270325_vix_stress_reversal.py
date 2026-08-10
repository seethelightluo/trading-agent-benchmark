import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); ret=close.pct_change();
# Macro stress-conditioned reversal: reverse yesterday's cross-sectional move only when VIX rose sharply.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close
vix=vix.reindex(close.index).ffill(); stress=(vix.pct_change()>vix.pct_change().rolling(20,min_periods=10).quantile(.6)).astype(float)
fac=-ret.shift(0).mul(stress,axis=0); fac.to_csv('scripts/miner_1_20270325_vix_stress_reversal_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h);vals=[];ns=[];ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));ds.append(dt)
 return pd.Series(vals,index=ds),ns
print('assets',len(C),'rows',len(fac),'active',int(stress.sum()))
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
