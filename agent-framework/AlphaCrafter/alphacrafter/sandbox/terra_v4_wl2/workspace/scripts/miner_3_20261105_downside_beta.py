import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-11-04')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date') for s in U}
# Market downside beta: estimate asset covariance with SPX only on SPX-down sessions;
# low downside beta is defensive and may diversify the existing price-only factors.
for w in [40,60,90]:
 spx=D['SPX'].close.pct_change(); rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); z=pd.concat([r,spx],axis=1,keys=['r','m']); down=z.m.where(z.m<0); cov=(z.r*down).rolling(w,min_periods=w//2).mean()-z.r.rolling(w,min_periods=w//2).mean()*down.rolling(w,min_periods=w//2).mean(); var=(down**2).rolling(w,min_periods=w//2).mean()-down.rolling(w,min_periods=w//2).mean()**2
  f=-cov/(var+1e-10); y=x.close.shift(-1)/x.close-1
  for dt in x.index:
   if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); z=[];ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.array(z); print('w',w,'dates',len(z),'avg_names',np.mean(ns),'coverage',a.s.nunique()/15,'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
 rr=a.assign(rk=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rk');print('turnover',rr.diff().abs().mean().mean())
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-04')]:
  q=[spearmanr(g.f,g.y).statistic for dt,g in a.groupby('date') if lo<=str(dt.date())<=hi and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1];q=np.array(q);print(lo[:4],len(q),'mean %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
 for h in [5,10]:
  q=[]
  for s,x in D.items():
   r=x.close.pct_change(); z2=pd.concat([r,spx],axis=1,keys=['r','m']); down=z2.m.where(z2.m<0); cov=(z2.r*down).rolling(w,min_periods=w//2).mean()-z2.r.rolling(w,min_periods=w//2).mean()*down.rolling(w,min_periods=w//2).mean();var=(down**2).rolling(w,min_periods=w//2).mean()-down.rolling(w,min_periods=w//2).mean()**2; f=-cov/(var+1e-10); y=x.close.shift(-h)/x.close-1
   q += [(dt,float(f.loc[dt]),float(y.loc[dt])) for dt in x.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
  b=pd.DataFrame(q,columns=['date','f','y']); ic=np.array([spearmanr(g.f,g.y).statistic for _,g in b.groupby('date') if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1]);print('h',h,'dates',len(ic),'IC %.6f ICIR %.6f'%(ic.mean(),ic.mean()/ic.std(ddof=1)))
