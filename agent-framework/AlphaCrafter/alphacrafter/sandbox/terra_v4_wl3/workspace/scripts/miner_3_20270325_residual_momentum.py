import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 px[a]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); ret=p.pct_change()
# beta-neutral relative momentum: 20d asset return less rolling 60d beta to equal-weight benchmark times benchmark return
bench=ret.mean(axis=1); cov=ret.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0); rb=ret.rolling(20,min_periods=15).sum(); bm=bench.rolling(20,min_periods=15).sum()
fac=rb-beta.mul(bm,axis=0)
fac.to_csv('scripts/miner_3_20270325_residual_momentum_signal.csv')
fwd=p.pct_change(1).shift(-1); vals=[]; ds=[]; ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
s=pd.Series(vals,index=ds)
print('assets',len(assets),'rows',len(fac),'dates',len(s),'avgN',np.mean(ns),'coverage',fac.notna().sum(axis=1).mean()/len(assets))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
for h in [5,10]:
 ff=p.pct_change(h).shift(-h); vv=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('H',h,'IC %.6f ICIR %.6f'%(np.mean(vv),np.mean(vv)/np.std(vv,ddof=1)))
