import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); root='../persistent/stock_data'; assets=[os.path.basename(x)[:-4] for x in glob.glob(root+'/*.csv')]
D={}
for a in assets:
 d=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 # Lagged downside-risk adjusted momentum: all inputs known at signal close, forecast starts next day.
 down=r.where(r<0,0).rolling(40,min_periods=20).std(); mom=d.close.pct_change(20)
 D[a]=(mom/(down*np.sqrt(20)+1e-8)).clip(-10,10)
fac=pd.DataFrame(D).sort_index(); fac.to_csv('scripts/miner_3_20270325_downside_momentum_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd={a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close.pct_change(h).shift(-h) for a in assets}; fw=pd.DataFrame(fwd).reindex(fac.index)
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
