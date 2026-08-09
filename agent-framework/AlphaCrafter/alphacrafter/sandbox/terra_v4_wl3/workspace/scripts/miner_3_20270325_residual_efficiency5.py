import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; fw={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); c=d.close; r=c.pct_change()
 eff=r.rolling(10,min_periods=10).sum()/r.abs().rolling(10,min_periods=10).sum()
 F[a]=c.pct_change(20)*eff
 for h in fw: fw[h][a]=c.pct_change(h).shift(-h)
raw=pd.DataFrame(F).sort_index()
# Remove contemporaneous cross-sectional exposure to short reversal (3d return); residual is still observable at date t.
rev={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); rev[a]=d.close.pct_change(5)
rev=pd.DataFrame(rev).reindex(raw.index)
res=pd.DataFrame(index=raw.index,columns=assets,dtype=float)
for dt in raw.index:
 x=raw.loc[dt]; y=rev.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].nunique()>1:
  b=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/np.var(z.iloc[:,1],ddof=1); res.loc[dt,z.index]=z.iloc[:,0]-b*z.iloc[:,1]
res.to_csv('scripts/miner_3_20270325_residual_efficiency5_signal.csv')
print('assets',len(assets),'rows',len(res),'period',res.index.min(),res.index.max())
for h in fw:
 fwd=pd.DataFrame(fw[h]).reindex(res.index); vals=[]; ds=[]; ns=[]
 for dt in res.index:
  z=pd.concat([res.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),res.notna().sum(axis=1).mean()/len(assets)))
 if h==1:
  for p in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=p[0])&(s.index<=p[1])]; print('regime',p,'IC %.6f n %d'%(q.mean(),len(q)))
print('turnover',res.rank(axis=1,pct=True).diff().abs().mean().mean())
