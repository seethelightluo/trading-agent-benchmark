"""2030-12-12 revalidation: inverse peer-relative directional consistency, 60 sessions.
One candidate only. Lag signal one completed day; daily cross-sectional Spearman IC.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']
def series(a):
    d=get_stock_daily_data(a,5000).copy()
    d['date']=pd.to_datetime(d['date']).dt.normalize()
    return pd.Series(pd.to_numeric(d['close'],errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:series(a) for a in assets}).sort_index()
R=P.pct_change()
F=(-np.sign(R.sub(R.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def stats(h,lo=None,hi=None):
    x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; breadth=[]
    for d in x.index:
        q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>2:
            v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(v): z.append(v); breadth.append(len(q))
    if not z:return {'ic_dates':0}
    z=np.asarray(z); return {'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit_ratio':round(float((z>0).mean()),4),'ic_dates':len(z),'mean_instruments':round(float(np.mean(breadth)),3),'minimum_instruments':min(breadth)}
cut=P.index.max()
print('FACTOR inverse_peer_relative_directional_consistency_60')
print('VALIDATION_DATE 2030-12-12 DATA_ENDPOINT',cut.date(),'UNIVERSE',len(assets),'CALENDAR_DATES',len(P))
print('COVERAGE_CELLS',int(F.notna().sum().sum()),'/',F.size,round(float(F.notna().stack().mean()),6))
for h in (1,5,10,20): print('HORIZON',h,stats(h))
for n,lo,hi in [('2025_2026','2025-01-01','2026-12-31'),('2027_2028','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent_180d',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME_10D',n,stats(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
print('NOVELTY_AUDIT_STATUS incomplete: all-current-library reconstruction did not finish within the 300s runtime limit; admission prohibited.')
