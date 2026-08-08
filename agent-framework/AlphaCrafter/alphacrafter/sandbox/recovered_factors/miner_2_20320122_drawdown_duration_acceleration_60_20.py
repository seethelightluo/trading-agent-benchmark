"""One idea: cross-asset drawdown-duration acceleration (60d reference, 20d transition)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; closes={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 closes[a]=pd.to_numeric(d.sort_values('date').set_index('date')['close'],errors='coerce')
p=pd.DataFrame(closes); r=p.pct_change()
# Number of consecutive completed sessions below a trailing 60-session peak.
under=p.lt(p.rolling(60,min_periods=45).max())
def duration(s):
 out=[]; n=0
 for z in s.fillna(False):
  n=n+1 if z else 0; out.append(n)
 return pd.Series(out,index=s.index,dtype=float)
dur=under.apply(duration)
# Negative increase in underwater duration: high scores identify drawdowns that are shortening / recovering.
sig=-(dur-dur.shift(20)); sig=sig.sub(sig.median(axis=1),axis=0).shift(1)
fwd={h:p.shift(-h).div(p).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=sig.loc[lo:hi] if lo else sig; z=[]; b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); b.append(len(q))
 z=np.asarray(z)
 return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(b),3),'min_breadth':min(b)}
cut=p.dropna(how='all').index.max()
rank=sig.rank(axis=1,pct=True)
print('FACTOR drawdown_duration_acceleration_60_20 CUTOFF',cut.date(),'ASSETS',len(assets))
print('CELLS',int(sig.notna().sum().sum()),'/',sig.size,'COVERAGE',round(sig.notna().stack().mean(),6),'TURNOVER',round(rank.diff().abs().stack().mean(),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cut.date())),('recent180',str(cut-pd.Timedelta(days=180)),str(cut.date()))]: print('REGIME10',n,stat(10,lo,hi))
# Structural diagnostics only; full admitted-library test follows only if gates pass.
dd=p.div(p.rolling(60,min_periods=45).max()).sub(1)
recovery=(dd-dd.shift(10)).div(.01-dd.shift(10))
vol=r.rolling(20,min_periods=15).std()
for n,x in {'smooth_drawdown_recovery_60_10':recovery,'risk_adjusted_trend_20d':p.pct_change(20).div(vol),'inverse_idiosyncratic_volatility_20':-vol}.items():
 q=pd.concat([sig.stack(),x.stack()],axis=1).dropna();print('PROXY',n,'cells',len(q),'rho',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6))
