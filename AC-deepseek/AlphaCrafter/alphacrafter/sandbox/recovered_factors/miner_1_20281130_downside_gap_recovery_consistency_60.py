# miner_1: downside gap-recovery consistency, a single price-path candidate
import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2028-11-29')
O={}; C={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cutoff]
 O[a]=pd.to_numeric(d['open'],errors='coerce'); C[a]=pd.to_numeric(d['close'],errors='coerce')
O=pd.DataFrame(O).sort_index(); P=pd.DataFrame(C).sort_index(); R=P.pct_change()
# On sessions opening below the prior close, measure recovery from open to close.
# Average conditional recovery over 60 sessions; require 12 events, then center cross-section.
gap=O.div(P.shift(1)).sub(1); intr=P.div(O).sub(1)
raw=intr.where(gap<0).rolling(60,min_periods=12).mean()
F=raw.sub(raw.median(axis=1),axis=0)
print('candidate downside_gap_recovery_consistency_60: mean close/open return conditional on open<prior close, 60 sessions, median-centered')
print('cutoff',cutoff.date(),'rows',len(F),'cells',int(F.notna().sum().sum()),'/',F.size,'coverage',round(F.notna().sum().sum()/F.size,4),'mean_names',round(F.notna().sum(axis=1).mean(),2))
ranks=F.rank(axis=1,pct=True); print('daily rank-change turnover',round(ranks.diff().abs().stack().mean(),6),'mean_cs_std',round(F.std(axis=1).mean(),6))
for h in [1,5,10,20]:
 fw=R.shift(-1).rolling(h,min_periods=h).apply(lambda x:np.prod(1+x)-1,raw=True).shift(-(h-1))
 out=[]; breadth=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): out.append((dt,v));breadth.append(len(z))
 x=np.array([v for _,v in out]); print(f'H{h}: dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} breadth={np.mean(breadth):.2f}')
 if h==10:
  for name,s,e in [('2025-26','2025-01-01','2026-12-31'),('2027-current','2027-01-01','2028-11-29'),('recent180','2028-06-02','2028-11-29')]:
   q=np.array([v for d,v in out if pd.Timestamp(s)<=d<=pd.Timestamp(e)])
   print(name,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),4) if len(q) else None)
