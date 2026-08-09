import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
raw={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); raw[a]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(raw).sort_index(); ret=px.pct_change()
r20=px.pct_change(20); med=r20.median(axis=1); cs=r20.sub(med,axis=0)
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
# rolling per column with NaN-tolerant time series, then cross-sectional scoring
fac=(cs/(vol+1e-8)).apply(lambda x:x.rolling(3,min_periods=2).mean())
fac.to_csv('scripts/miner_1_20270325_relative_trend_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(dt); ns.append(len(q))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  print('regimes',[(x,len(s[(s.index>=lo)&(s.index<=hi)]),round(s[(s.index>=lo)&(s.index<=hi)].mean(),6)) for x,lo,hi in [('20-22','2020','2022-12-31'),('23-24','2023','2024-12-31'),('25-27','2025','2027-03-24')]])
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
