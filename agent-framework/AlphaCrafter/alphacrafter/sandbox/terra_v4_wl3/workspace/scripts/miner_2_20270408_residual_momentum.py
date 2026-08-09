import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); base='../persistent/stock_data/'
assets=[os.path.basename(x)[:-4] for x in glob.glob(base+'*.csv')]
P={}
for a in assets:
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).sort_values('date'); P[a]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# Idiosyncratic momentum: 20d return residual to rolling 60d beta on SPX, damped by residual volatility.
bench=r['SPX']; cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0); resid=r.sub(beta.mul(bench,axis=0),axis=0)
raw=resid.rolling(20,min_periods=15).sum()/(resid.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-8)
raw.to_csv('scripts/miner_2_20270408_residual_momentum_signal.csv')
print('assets',len(assets),'rows',len(raw),'period',raw.index.min(),raw.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in raw.index:
  z=pd.concat([raw.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',raw.notna().sum(axis=1).mean()/len(assets),'turnover',raw.rank(axis=1,pct=True).diff().abs().mean().mean())
