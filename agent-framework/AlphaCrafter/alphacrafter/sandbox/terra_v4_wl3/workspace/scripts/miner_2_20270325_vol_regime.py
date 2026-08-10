import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; fw={1:{},5:{},10:{}}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
 # Volatility regime factor: prefer persistent low risk, but reward assets whose recent vol is falling.
 F[a]=(-v20/(v60.replace(0,np.nan)) + 0.5*(1-v20/(v60.replace(0,np.nan)))).where(v60.notna())
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_vol_regime_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(fac.index); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
