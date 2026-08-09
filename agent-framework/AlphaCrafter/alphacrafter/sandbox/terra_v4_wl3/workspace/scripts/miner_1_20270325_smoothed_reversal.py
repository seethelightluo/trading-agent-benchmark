import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 px[a]=d[d.date<=cut].set_index('date')['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vol=r.rolling(20,min_periods=10).std()
z=(r/(vol+1e-12)).clip(-5,5)
fac=-(0.5*z+0.3*z.shift(1)+0.2*z.shift(2))
fac=fac.sub(fac.median(axis=1),axis=0)
fac=fac.clip(lower=fac.quantile(.05,axis=1),upper=fac.quantile(.95,axis=1),axis=0)
fac.to_csv('scripts/miner_1_20270325_smoothed_reversal_signal.csv',index_label='date')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
allstats={}
for h in [1,5,10]:
 fwd=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
 s=pd.Series(vals)
 ic=float(s.mean()); icir=float(ic/s.std(ddof=1)); allstats[h]=(len(s),float(np.mean(ns)),ic,icir,float((s>0).mean()))
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(ic,icir,(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s # regime summary omitted index alignment
 print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
print(json.dumps(allstats))
