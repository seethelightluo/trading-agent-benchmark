import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
R={}; V={}; C={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=cut].set_index('date'); c=d.close; r=c.pct_change(); C[a]=c;R[a]=r.rolling(3,min_periods=3).sum();V[a]=r.rolling(20,min_periods=20).std()
idx=sorted(set().union(*[x.index for x in C.values()])); r3=pd.DataFrame(R).reindex(idx); vol=pd.DataFrame(V).reindex(idx); close=pd.DataFrame(C).reindex(idx)
f=-(r3-r3.median(axis=1,skipna=True))/(vol*np.sqrt(3)+1e-12);f.to_csv('scripts/miner_2_20270325_idio_reversal_signal.csv')
print('assets',len(assets),'rows',len(f),'period',f.index[0],f.index[-1])
for h in [1,5,10]:
 fw={a:C[a].pct_change(h).shift(-h) for a in assets};fw=pd.DataFrame(fw).reindex(idx);vals=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',f.notna().sum(axis=1).mean()/len(assets),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
