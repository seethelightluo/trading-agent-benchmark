"""Operational audit of admitted relative-volume factor; exact persisted formula."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); V=pd.DataFrame(V).reindex(P.index); cutoff=P.dropna(how='all').index.max()
# Exact definition from miner_3_20260716 JSON: own 20-observation arithmetic mean,
# at least 15 observations, natural logarithm. Centre only for IC comparability;
# centering does not alter a same-date Spearman rank.
raw=np.log(V/V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
sig=raw.sub(raw.median(axis=1),axis=0)
fw=P.shift(-10)/P-1
ics=[];breadth=[]
for dt in sig.index:
 q=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): ics.append(z);breadth.append(len(q))
z=np.array(ics)
print('FACTOR_AUDIT relative_volume_participation_20d CUTOFF',cutoff.date(),'ASSETS',len(A))
print('RAW_CELLS',int(raw.notna().sum().sum()),'TOTAL',raw.size,'COVERAGE',round(float(raw.notna().stack().mean()),6))
print('UNIQUE_PER_ASSET', {a:int(raw[a].dropna().nunique()) for a in A})
print('CROSS_SECTION_NONCONSTANT_DATES',int((sig.nunique(axis=1)>1).sum()),'VALID_IC_DATES',len(z),'MEAN_BREADTH',round(float(np.mean(breadth)),3) if breadth else 0,'MIN_BREADTH',min(breadth) if breadth else 0)
print('IC10',round(float(z.mean()),6) if len(z) else None,'ICIR10',round(float(z.mean()/z.std(ddof=1)),6) if len(z)>1 else None,'HIT',round(float((z>0).mean()),6) if len(z) else None)
# Evidence needed by the full-library audit is a pairwise Spearman over factor cells.
# Report both raw and median-centred versions, and explicitly test finite variance.
for label,x in [('raw',raw),('centred',sig)]:
 q=x.stack().dropna(); print(label,'PAIR_CELLS',len(q),'GLOBAL_UNIQUE',q.nunique(),'STD',round(float(q.std()),10),'FINITE_VARIANCE',bool(q.nunique()>1 and q.std()>0))
print('LATEST_SIGNAL',sig.loc[:cutoff].tail(1).round(8).to_dict('records')[0])
