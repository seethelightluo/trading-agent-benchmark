"""Single-idea validation: 3-observation smoothed abnormal-range reversal.
The prior one-day range-expansion reversal was predictive but noisy.  This variant
averages the completed daily exhaustion score over three native observations to
reduce transitory ranking changes while retaining its short-horizon interpretation.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-01'); HS=[1,5,10,20]
close={}; sigs={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 for x in ['high','low','close']: d[x]=pd.to_numeric(d[x],errors='coerce')
 c=d.close.replace(0,np.nan); ret=c.pct_change(fill_method=None)
 rr=((d.high-d.low).abs()/c).replace([np.inf,-np.inf],np.nan)
 baseline=rr.rolling(20,min_periods=8).median().shift(1)
 raw=(-ret*(rr/baseline)).replace([np.inf,-np.inf],np.nan)
 sigs[a]=raw.rolling(3,min_periods=2).mean()
 close[a]=c
close=pd.DataFrame(close); sig=pd.DataFrame(sigs).reindex(close.index)
print('FACTOR smoothed_abnormal_range_reversal_3_20obs cutoff',END.date(),'assets',len(A))
print('cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,6),'mean_assets_per_date',round(sig.notna().sum(axis=1).mean(),3))
for h in HS:
 y=close.pct_change(h,fill_method=None).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); dates.append(dt); ns.append(len(q))
 x=np.array(vals); ds=pd.DatetimeIndex(dates); n=np.array(ns); ir=x.mean()/x.std(ddof=1)
 print('H',h,'ic_dates',len(x),'daily_paper_IC',round(x.mean(),6),'ICIR',round(ir,6),'hit',round((x>0).mean(),5),'mean_instruments',round(n.mean(),3),'min_instruments',n.min(),'GATE',('PASS' if abs(x.mean())>=.007 and abs(ir)>=.084 else 'FAIL'))
 for lab,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_30','2026-01-01','2030-05-01')]:
  v=x[(ds>=lo)&(ds<=hi)]
  if len(v)>1: print(' REGIME',h,lab,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),5))
rk=sig.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(np.abs(q.iloc[:,0]-q.iloc[:,1]).mean())
print('rank_turnover',round(float(np.mean(turns)),6),'adjacent_dates',len(turns))
sig.to_pickle('scripts/miner_1_20300502_smoothed_abnormal_range_reversal_3_20obs_candidate_signal.pkl')
