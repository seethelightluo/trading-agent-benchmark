"""miner_1: conditional intraday downside recovery strength using completed native OHLC bars."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-02-06'); HS=(1,5,10,20)
F={};Y={h:{} for h in HS}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 c=d.close.astype(float); o=d.open.astype(float).replace(0,np.nan); r=c.pct_change(fill_method=None)
 # On a net-down day, intraday return captures whether the asset recovered from a worse open.
 # Average it over completed 60 native observations; min 15 qualifying downside days.
 intraday=c/o-1
 f=intraday.where(r<0).rolling(60,min_periods=15).mean()
 F[a]=f
 for h in HS:Y[h][a]=(1+r).rolling(h,min_periods=h).apply(np.prod,raw=True).shift(-h)-1
sig=pd.DataFrame(F); print('FACTOR downside_day_intraday_recovery_60obs cutoff',END.date(),'assets',len(A))
print('cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,6),'mean_assets_per_date',round(sig.notna().sum(axis=1).mean(),3))
def assess(h):
 y=pd.DataFrame(Y[h]); z=[]; ds=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(dt);ns.append(len(q))
 return np.array(z),pd.DatetimeIndex(ds),np.array(ns)
R={}
for h in HS:
 x,ds,n=assess(h);R[h]=(x,ds,n); print('H',h,'ic_dates',len(x),'daily_paper_IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5),'mean_instruments',round(n.mean(),3),'min_instruments',n.min())
x,ds,n=R[5]
for lab,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_30','2026-01-01','2030-02-06')]:
 z=x[(ds>=lo)&(ds<=hi)]; print('REGIME',lab,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),5))
rk=sig.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('rank_turnover',round(float(np.mean(ts)),6),'turnover_dates',len(ts))
for h,(x,_,_) in R.items(): print('GATE_H',h,'PASS' if abs(x.mean())>=.007 and abs(x.mean()/x.std(ddof=1))>=.084 else 'FAIL')
sig.to_pickle('scripts/miner_1_20300207_downside_day_intraday_recovery_60obs_signal.pkl')
