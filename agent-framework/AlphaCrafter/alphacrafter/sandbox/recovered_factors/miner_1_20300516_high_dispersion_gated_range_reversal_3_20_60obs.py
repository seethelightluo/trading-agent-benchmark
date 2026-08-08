"""Single idea: high-dispersion-gated smoothed abnormal-range reversal.
Cross-asset dispersion identifies stressed, heterogeneous sessions where a large
completed intraday range combined with a directional move is more likely to be
short-term exhaustion than trend continuation. The signal is intentionally only
available in a pre-established elevated-dispersion regime; all inputs are lagged
or from the completed bar.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-15'); HS=[1,5,10,20]
close={}; raw={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 for x in ['high','low','close']: d[x]=pd.to_numeric(d[x],errors='coerce')
 c=d.close.replace(0,np.nan); r=c.pct_change(fill_method=None)
 rr=((d.high-d.low).abs()/c).replace([np.inf,-np.inf],np.nan)
 normal=rr.rolling(20,min_periods=8).median().shift(1)
 raw[a]=(-r*(rr/normal)).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=2).mean()
 close[a]=c
close=pd.DataFrame(close); base=pd.DataFrame(raw).reindex(close.index)
# Completed-day cross-sectional dispersion; regime cutoff is strictly lagged.
rets=close.pct_change(fill_method=None)
disp=rets.median(axis=1).sub(rets,axis=1).abs().median(axis=1)
pctl=disp.rolling(60,min_periods=30).rank(pct=True).shift(1)
active=pctl>=0.70
sig=base.where(active, np.nan)
print('FACTOR high_dispersion_gated_smoothed_abnormal_range_reversal_3_20_60obs cutoff',END.date(),'assets',len(A))
print('regime_active_dates',int(active.sum()),'/',len(active),round(active.mean(),4))
print('cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,6),'mean_assets_per_date',round(sig.notna().sum(axis=1).mean(),3))
for h in HS:
 y=close.pct_change(h,fill_method=None).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); dates.append(dt); ns.append(len(q))
 x=np.array(vals); ds=pd.DatetimeIndex(dates); n=np.array(ns)
 ir=x.mean()/x.std(ddof=1) if len(x)>1 else np.nan
 print('H',h,'ic_dates',len(x),'daily_paper_IC',round(x.mean(),6),'ICIR',round(ir,6),'hit',round((x>0).mean(),5),'mean_instruments',round(n.mean(),3),'min_instruments',n.min(),'GATE',('PASS' if abs(x.mean())>=.007 and abs(ir)>=.084 else 'FAIL'))
 for lab,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_30','2026-01-01','2030-05-15')]:
  v=x[(ds>=lo)&(ds<=hi)]
  if len(v)>1: print(' REGIME',h,lab,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),5))
rk=sig.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(np.abs(q.iloc[:,0]-q.iloc[:,1]).mean())
print('rank_turnover',round(float(np.mean(turns)),6) if turns else None,'adjacent_dates',len(turns))
print('decay_interpretation: IC at 1,5,10,20 completed-day forward horizons printed above')
sig.to_pickle('scripts/miner_1_20300516_high_dispersion_gated_range_reversal_candidate_signal.pkl')
