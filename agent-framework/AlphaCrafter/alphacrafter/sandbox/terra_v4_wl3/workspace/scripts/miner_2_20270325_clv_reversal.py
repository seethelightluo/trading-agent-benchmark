import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; R={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date')
 rng=(d.high-d.low).replace(0,np.nan); clv=((2*d.close-d.high-d.low)/rng).clip(-1,1)
 F[a]=-clv.rolling(5,min_periods=3).mean(); R[a]=d.close.pct_change().shift(-1)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_clv_reversal_signal.csv'); fw=pd.DataFrame(R).reindex(fac.index)
def run(fw):
 v=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 return pd.Series(v,index=ds),ns
s,n=run(fw); print('assets',len(assets),'dates',len(s),'avgN',round(np.mean(n),2),'IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.6f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.notna().sum(axis=1).mean()/len(assets),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f ICIR %.6f dates %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
for h in [5,10]:
 ss,_=run(pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.pct_change(h).shift(-h) for a in assets}).reindex(fac.index));print('horizon',h,'IC %.6f ICIR %.6f dates %d'%(ss.mean(),ss.mean()/ss.std(ddof=1),len(ss)))
