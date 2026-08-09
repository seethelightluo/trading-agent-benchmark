"""Miner 2: one candidate -- inverse peer-relative directional consistency (60).
At each completed session, take an asset return less the contemporaneous 15-asset
median return, convert it to a direction (+1/-1/0), and average 60 sessions.
The candidate is the NEGATIVE of this persistence score: persistent peer-relative
winners are hypothesized to reverse. Inputs are lagged one day before outcomes.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']
def prices(a):
    d=get_stock_daily_data(a,5000).copy()
    d['date']=pd.to_datetime(d['date']).dt.normalize()
    return pd.Series(pd.to_numeric(d['close'],errors='coerce').to_numpy(),index=d['date']).groupby(level=0).last()
P=pd.DataFrame({a:prices(a) for a in assets}).sort_index()
R=P.pct_change()
relative=R.sub(R.median(axis=1),axis=0)
direction=np.sign(relative)
# Need 45 valid observations, deliberately avoiding a score driven by sparse starts.
F=(-direction.rolling(60,min_periods=45).mean()).shift(1)
F=F.sub(F.median(axis=1),axis=0)
cut=P.index.max()

def ic_stats(h,lo=None,hi=None):
    x=F.loc[lo:hi]; y=(P.shift(-h).div(P)-1).reindex(x.index)
    vals=[]; ns=[]
    for dt in x.index:
        q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>2:
            v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(v): vals.append(v); ns.append(len(q))
    if not vals:return {'ic_dates':0}
    z=np.array(vals); sd=z.std(ddof=1)
    return {'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/sd),6),
            'hit_ratio':round(float((z>0).mean()),4),'ic_dates':len(z),
            'mean_instruments':round(float(np.mean(ns)),3),'minimum_instruments':int(min(ns))}
print('FACTOR inverse_peer_relative_directional_consistency_60')
print('VALIDATION_ENDPOINT',cut.date(),'UNIVERSE',len(assets),'CALENDAR_DATES',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6))
for h in [1,5,10,20]:print('HORIZON',h,ic_stats(h))
for name,lo,hi in [('2020_2022','2020-01-01','2022-12-31'),('2023_2024','2023-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2028','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent_180_calendar_days',str(cut-pd.Timedelta(days=180)),None)]:
 print('REGIME_10D',name,ic_stats(10,lo,hi))
print('TURNOVER_MEAN_DAILY_RANK_CHANGE',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
print('MEAN_CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('NOTE library novelty audit intentionally deferred unless aggregate gates pass.')
