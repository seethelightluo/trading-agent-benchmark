import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut='2027-02-24'; px={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 d=pd.read_csv(p,usecols=['date','close']);d.date=pd.to_datetime(d.date);px[os.path.basename(p)[:-4]]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv',usecols=['date','close']);v.date=pd.to_datetime(v.date);v=v[v.date<=cut].set_index('date').close
r=np.log(p).diff(); vr=np.log(v).diff(); beta=r.rolling(60).cov(vr).div(vr.rolling(60).var(),axis=0)
f=-beta; fr=p.pct_change().shift(-1);a=[];rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):a.append(q);rows.append((dt,q,len(z)))
a=np.array(a);print('dates',len(a),'mean_ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'n',np.mean([n for _,_,n in rows]))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-24')]:
 q=[x for d,x,n in rows if str(d.date())>=lo and str(d.date())<=hi];print(lo,len(q),np.mean(q) if q else None,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else None)
out=pd.DataFrame(f).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal');out.to_csv('../persistent/factor_signals_miner_2_20270226_vixbeta.csv',index=False)
