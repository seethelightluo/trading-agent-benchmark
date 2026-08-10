import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
    a=os.path.basename(p)[:-4]
    d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
    C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index()
# Volatility-scaled 3-day cross-sectional reversal, active during elevated dispersion.
r3=close.pct_change(3)
vol=close.pct_change().rolling(20,min_periods=10).std()
raw=-r3/vol.replace(0,np.nan)
# cross-sectional rank-like centered score avoids scale differences
fac=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1).replace(0,np.nan),axis=0)
disp=close.pct_change().std(axis=1)
threshold=disp.rolling(60,min_periods=30).median()
fac=fac.where(disp.gt(threshold),np.nan)
fac.to_csv('scripts/miner_2_20270325_volscaled3_reversal_signal.csv')
def evaluate(h):
 y=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); dates.append(dt)
 s=pd.Series(vals,index=dates)
 return s,ns
print('assets',len(C),'rows',len(fac),'signal_dates',fac.notna().any(axis=1).sum())
for h in [1,5,10]:
 s,n=evaluate(h)
 print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]
   print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
