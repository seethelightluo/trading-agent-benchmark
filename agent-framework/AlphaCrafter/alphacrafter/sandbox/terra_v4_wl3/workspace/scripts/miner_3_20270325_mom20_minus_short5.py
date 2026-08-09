import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; R={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 vol=r.rolling(20,min_periods=15).std()
 # medium-term continuation while avoiding short-term crowded move; volatility-scaled
 F[a]=((d.close.pct_change(20)-0.5*d.close.pct_change(5))/vol).clip(-10,10)
 R[a]=d.close.pct_change()
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20270325_mom20_minus_short5_signal.csv')
rets=pd.DataFrame(R).reindex(fac.index)
out={}
for h in [1,5,10]:
 fwd=rets.rolling(h).sum().shift(-h+1) if h>1 else rets.shift(-1)
 vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds); out[h]=s
 print('h',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'coverage',round(fac.notna().sum(axis=1).mean()/len(assets),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]: print('regime',lo,hi,round(s[(s.index>=lo)&(s.index<=hi)].mean(),6))
print('saved signal rows',len(fac),'assets',len(assets))
