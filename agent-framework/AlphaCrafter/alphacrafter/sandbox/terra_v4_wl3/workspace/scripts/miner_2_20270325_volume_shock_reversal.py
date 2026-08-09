import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; px={}; vol={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); px[a]=d.close; vol[a]=d.volume
idx=pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in px.values()]))); F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 p=px[a]; r=p.pct_change(); v=np.log(vol[a].replace(0,np.nan)); shock=v-v.rolling(20,min_periods=10).median(); F[a]=(-r*shock).clip(-.15,.15)
 for h in fw: fw[h][a]=p.pct_change(h).shift(-h)
fac=pd.DataFrame(F).reindex(idx).sort_index(); fac.to_csv('scripts/miner_2_20270325_volume_shock_reversal_signal.csv')
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(fac.index); vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=pd.DatetimeIndex(ds)); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'IC %.6f n %d'%(q.mean(),len(q)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
