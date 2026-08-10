import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Trend acceleration: recent 20d return versus the preceding 40d return, rewarding improving relative trend.
fac=r.rolling(20,min_periods=20).sum()-r.shift(20).rolling(40,min_periods=40).sum()/2
fac.to_csv('scripts/miner_1_20270327_trend_acceleration_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h);v=[];n=[];ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z));ds.append(dt)
 return pd.Series(v,index=ds),n
print('assets',len(C),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 s,n=ev(h);print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
