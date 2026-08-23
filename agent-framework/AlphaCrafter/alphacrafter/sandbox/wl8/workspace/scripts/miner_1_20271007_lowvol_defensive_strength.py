import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in ASSETS:
 d=pd.read_csv(os.path.join(base,a+'.csv')); d['date']=pd.to_datetime(d.date); px[a]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index()
# Defensive relative strength: lagged 10d relative return, rewarded when asset volatility is below cross-sectional median.
r=p.pct_change(10); rel=r.sub(r.median(axis=1),axis=0)
vol=p.pct_change().rolling(20,min_periods=15).std(); vrel=vol.div(vol.median(axis=1),axis=0)
f=(rel*(1/(1+vrel))).rolling(3,min_periods=3).mean().shift(1)
fr=p.pct_change().shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
rw=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=rw.ic
print('candidate=lowvol_defensive_relative_strength'); print('dates',len(rw),'rows',int(rw.n.sum()),'avg_n',rw.n.mean(),'coverage',rw.n.sum()/(len(rw)*15)); print('IC',ic.mean(),'ICIR_daily',ic.mean()/ic.std(ddof=1),'ICIR_annualized',ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit',(ic>0).mean())
for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026','2026-01-01','2026-12-31'),('2027','2027-01-01','2027-10-06'),('recent90','2027-06-01','2027-10-06')]:
 q=ic.loc[lo:hi]; print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('turnover_proxy',rank.diff().abs().mean(axis=1).mean())
for h in [5,10]:
 vals=[]
 fy=p.pct_change(h).shift(-1)
 for dt in f.index:
  z=pd.concat([f.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('horizon',h,'dates',len(a),'IC',a.mean(),'ICIR_daily',a.mean()/a.std(ddof=1))
