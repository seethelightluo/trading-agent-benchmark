import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
files=glob.glob('../persistent/stock_data/*.csv'); names=[os.path.basename(p)[:-4] for p in files]
C={}
for p in files:
 n=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[n]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Volatility-scaled, cross-sectionally relative medium-term trend: 20d return
# relative to the daily cross-sectional median, divided by trailing 20d realized vol.
ret20=close.pct_change(20); rel=ret20.sub(ret20.median(axis=1),axis=0)
vol=r.rolling(20,min_periods=12).std()*np.sqrt(20)
fac=(rel/vol.replace(0,np.nan)).clip(-10,10)
fac.to_csv('scripts/miner_3_20270325_vol_scaled_trend_signal.csv')
print('assets',len(names),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ds.append(dt); ns.append(len(x))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(names),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
