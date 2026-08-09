"""One idea: peer-relative range compression breakout (10/60 sessions).
The signal is inverse short/long realized-volatility ratio: assets whose recent
10-session volatility is compressed versus their own 60-session baseline may
have a favorable next-horizon expansion/trend profile.  It is ranked only
across the 15 tradable assets and lagged one completed session.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; close={}
for a in A:
    d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
    close[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(close).sort_index(); R=P.pct_change(); cutoff=P.dropna(how='all').index.max()
short=R.rolling(10,min_periods=8).std(); long=R.rolling(60,min_periods=45).std()
# Low short-versus-long volatility = compression; ranks preserve cross-asset comparability.
raw=(-short/long.replace(0,np.nan)).rank(axis=1,pct=True)
cand=(raw-raw.median(axis=1).values[:,None]).clip(-.5,.5).shift(1)
fw={h:P.shift(-h).div(P)-1 for h in (1,5,10,20)}
def st(h,lo=None,hi=None):
    x=cand.loc[lo:hi] if lo is not None else cand; vals=[]; breadth=[]
    for dt in x.index:
        q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
        if len(q)>=8:
            z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(z): vals.append(z); breadth.append(len(q))
    vals=np.array(vals)
    return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR peer_relative_volatility_compression_10_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20): print('H',h,st(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]: print('REGIME10',n,st(10,lo,hi))
print('ADMISSION_NOTE: library signal correlation audit follows only a passed paper IC/ICIR gate.')
