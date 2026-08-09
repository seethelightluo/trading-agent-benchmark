import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; H={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 F[a]=((d.close.pct_change(20)-0.5*d.close.pct_change(5))/vol).replace([np.inf,-np.inf],np.nan).clip(-10,10)
 for h in H: H[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); print('factor medium_momentum_short_reversal')
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for h in [1,5,10]:
 fwd=pd.DataFrame(H[h]).reindex(fac.index); vals=[]; dates=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates); print(h,'d dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
  q=s[(s.index>=lo)&(s.index<=hi)]; print(' regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
fac.to_csv('scripts/miner_2_20270325_medium_momentum_short_reversal_signal.csv')
