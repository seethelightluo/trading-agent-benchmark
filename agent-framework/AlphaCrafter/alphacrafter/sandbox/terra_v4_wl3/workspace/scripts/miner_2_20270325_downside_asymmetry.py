import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); base='../persistent/stock_data/'
assets=[os.path.basename(x)[:-4] for x in glob.glob(base+'*.csv')]
F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 # downside asymmetry: reward medium momentum when gains dominate losses, penalize downside-risk exposure
 down=r.clip(upper=0).pow(2).rolling(20,min_periods=20).mean().pow(.5)
 up=r.clip(lower=0).pow(2).rolling(20,min_periods=20).mean().pow(.5)
 asym=(up-down)/(up+down+1e-8)
 F[a]=r.rolling(20,min_periods=20).sum()*asym
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
raw=pd.DataFrame(F).sort_index(); raw.to_csv('scripts/miner_2_20270325_downside_asymmetry_signal.csv')
print('assets',len(assets),'rows',len(raw),'period',raw.index.min(),raw.index.max())
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(raw.index); vals=[]; ds=[]; ns=[]
 for dt in raw.index:
  z=pd.concat([raw.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),raw.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('turnover',raw.rank(axis=1,pct=True).diff().abs().mean().mean())
# decay by correlations of adjacent signal ranks
print('valid dates',raw.notna().any(axis=1).sum())
