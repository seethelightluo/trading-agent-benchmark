import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); base='../persistent/stock_data/'
assets=[os.path.basename(x)[:-4] for x in glob.glob(base+'*.csv')]
px={}; vol={}
for a in assets:
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 px[a]=d.close; vol[a]=d.volume
close=pd.DataFrame(px).sort_index(); volume=pd.DataFrame(vol).reindex(close.index)
r=close.pct_change(); lv=np.log(volume.replace(0,np.nan))
shock=lv-lv.rolling(20,min_periods=10).median()
# Smooth volume-shock reversal: recent return reversal weighted by abnormal log-volume, averaged over 3 sessions.
raw=(-r*shock).clip(-.15,.15)
fac=raw.rolling(3,min_periods=2).mean()
fac=fac.sub(fac.median(axis=1),axis=0)
fac.to_csv('scripts/miner_2_20270325_volume_shock_smooth_signal.csv')
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=pd.DatetimeIndex(ds)); print('H',h,'dates',len(s),'avgN %.2f IC %.7f ICIR %.7f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'n',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('assets',len(assets),'coverage %.6f turnover %.6f'%(fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
