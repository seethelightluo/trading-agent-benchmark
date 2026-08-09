import pandas as pd,numpy as np,os,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); A=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.pct_change()
for smooth in [5,10,20]:
 F={}; FW={}
 for a in A:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=cut]; r=d.close.pct_change()
  x=pd.concat([r.rename('r'),v.rename('v')],axis=1).dropna(); b=-x.r.rolling(60,min_periods=30).cov(x.v)/(x.v.rolling(60,min_periods=30).var()+1e-10)
  F[a]=b.rolling(smooth,min_periods=1).mean().reindex(d.index); FW[a]=d.close.pct_change().shift(-1)
 raw=pd.DataFrame(F); fwd=pd.DataFrame(FW).reindex(raw.index); ic=[]; ns=[]
 for dt in raw.index:
  z=pd.concat([raw.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(ic); print('smooth',smooth,'dates',len(s),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),raw.rank(pct=True).diff().abs().mean().mean()))
