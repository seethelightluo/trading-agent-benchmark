import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
rows=[]
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(fill_method=None); lv=np.log(d.volume.replace(0,np.nan)); shock=(lv-lv.rolling(20,min_periods=10).mean())/(lv.rolling(20,min_periods=10).std()+1e-8)
 # reversal is stronger after unusually high volume, but cap shock to limit outliers
 fac=(-r*shock.clip(-3,3)).rename('fac'); f1=d.close.pct_change(-1,fill_method=None);f5=d.close.pct_change(-5,fill_method=None);f10=d.close.pct_change(-10,fill_method=None)
 rows.append(pd.DataFrame({'asset':a,'fac':fac,'f1':f1,'f5':f5,'f10':f10}))
x=pd.concat(rows); print('assets',len(assets),'period',x.index.min(),x.index.max())
for h,col in [(1,'f1'),(5,'f5'),(10,'f10')]:
 vals=[];ds=[];ns=[]
 for dt,g in x.groupby(level=0):
  q=g[['fac',col]].dropna()
  if len(q)>=8 and q.fac.nunique()>1 and q[col].nunique()>1: vals.append(spearmanr(q.fac,q[col]).statistic);ds.append(dt);ns.append(len(q))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',x.fac.notna().mean())
x.reset_index().pivot(index='date',columns='asset',values='fac').to_csv('scripts/miner_1_20270325_volume_shock_signal.csv')
