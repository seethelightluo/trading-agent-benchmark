import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px={}; rr={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); px[a]=d.close; rr[a]=d.close.pct_change()
px=pd.DataFrame(px).sort_index(); rr=pd.DataFrame(rr).reindex(px.index); b=rr['SPX']; w=60
bm=b.rolling(w,min_periods=30).mean(); bv=((b-bm)**2).rolling(w,min_periods=30).mean()
beta=pd.DataFrame(index=rr.index,columns=assets,dtype=float)
for a in assets:
 xm=rr[a].rolling(w,min_periods=30).mean(); cov=((rr[a]-xm)*(b-bm)).rolling(w,min_periods=30).mean(); beta[a]=cov/bv
asset20=px.pct_change(20); b20=(1+b).rolling(20,min_periods=10).apply(np.prod,raw=True)-1
resid=rr.sub(beta.mul(b,axis=0),axis=0); rv=resid.rolling(20,min_periods=10).std()
fac=(asset20-beta*b20).div(rv*np.sqrt(20)).clip(-10,10); fac.to_csv('scripts/miner_2_20270325_residual_momentum_signal.csv')
print('assets',len(assets),'dates',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=dates); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
