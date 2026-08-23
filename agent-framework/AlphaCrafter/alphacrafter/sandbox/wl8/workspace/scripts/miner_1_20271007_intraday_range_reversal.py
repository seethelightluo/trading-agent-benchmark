import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; ds={}
for a in A:
 d=pd.read_csv(os.path.join(base,a+'.csv')); d.date=pd.to_datetime(d.date); ds[a]=d.sort_values('date').drop_duplicates('date').set_index('date')
o=pd.concat({a:ds[a].open for a in A},axis=1).sort_index(); c=pd.concat({a:ds[a].close for a in A},axis=1).sort_index(); h=pd.concat({a:ds[a].high for a in A},axis=1).sort_index(); l=pd.concat({a:ds[a].low for a in A},axis=1).sort_index()
# lagged intraday excursion reversal: yesterday's close location in range, smoothed 2d; no future data
clv=((c-l)/(h-l).replace(0,np.nan)-0.5).rolling(2,min_periods=2).mean().shift(1); f=-clv; fr=c.pct_change().shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=r.ic
print('candidate=intraday_range_reversal_2d'); print('dates',len(r),'rows',int(r.n.sum()),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('IC',x.mean(),'ICIR_daily',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for nm,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026','2026-01-01','2026-12-31'),('2027','2027-01-01','2027-10-06'),('recent90','2027-06-01','2027-10-06')]:
 q=x.loc[lo:hi]; print(nm,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
