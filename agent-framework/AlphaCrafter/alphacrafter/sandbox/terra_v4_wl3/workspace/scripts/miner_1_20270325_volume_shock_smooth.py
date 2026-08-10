import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); lv=np.log(d.volume.replace(0,np.nan)); shock=lv-lv.rolling(20,min_periods=10).median()
 F[a]=(-r*shock).rolling(3,min_periods=2).mean().clip(-.15,.15)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_1_20270325_volume_shock_smooth_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=fac.index.to_series().map(lambda x:x) # placeholder
 close=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}).sort_index().reindex(fac.index)
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
