import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2028-09-06')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=cutoff]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
mom=p.pct_change(20); down=r.where(r<0).rolling(60,min_periods=30).std(); f=(mom/(down+1e-8)).shift(1)
for k in [1,5,10,20]:
 res=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],p.pct_change(k).shift(-k).loc[dt]],axis=1).dropna()
  if len(z)>=8: res.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(res); print('horizon',k,'dates',len(q),'avg_names',round(float(pd.concat([f,p.pct_change(k).shift(-k)],axis=1).groupby(level=0).count().iloc[:,0].replace(0,np.nan).mean()),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
q=np.array([])
res=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],p.pct_change(1).shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: res.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
q=np.array(res); print('recent250 daily IC',q[-250:].mean(),'ICIR',q[-250:].mean()/q[-250:].std(ddof=1),'coverage',np.mean(ns)/15,'min_names',min(ns),'max_names',max(ns),'dates',len(q),'range',f.index.min().date(),f.index.max().date())
ranks=f.rank(axis=1,pct=True); tv=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: tv.append(np.abs(z.iloc[:,0]-z.iloc[:,1]).mean())
print('turnover_rank_change',np.mean(tv),'n_turnover_obs',len(tv))
