"""One idea: inverse downside close-location, a short-horizon mean-reversion signal."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; frames=[]
for s in assets:
 d=get_stock_daily_data(s,3000).copy(); d['date']=pd.to_datetime(d.date)
 # Close location in day's range. Restrict to down sessions so it captures selling-pressure exhaustion.
 rng=(d.high-d.low).replace(0,np.nan); clv=(2*d.close-d.high-d.low)/rng
 ret=d.close.pct_change(); down=ret<0
 # Candidate is NEGATIVE average close location on down days, trailing 20, then lag one day.
 d[s]=(-clv.where(down).rolling(20,min_periods=8).mean()).shift(1)
 frames.append(d.set_index('date')[[s]])
x=pd.concat(frames,axis=1).sort_index(); closes=[]
for s in assets:
 d=get_stock_daily_data(s,3000).copy();d.date=pd.to_datetime(d.date);closes.append(d.set_index('date').close.rename(s))
c=pd.concat(closes,axis=1).reindex(x.index)
print('candidate: -mean(CLV_t | return_t<0, trailing 20 sessions), lagged one session')
print('calendar',x.index.min().date(),x.index.max().date(),'dates',len(x),'cells valid',int(x.notna().sum().sum()),'/',x.size, 'coverage',x.notna().mean().mean())
for h in [1,5,10,20]:
 fwd=c.shift(-h)/c-1; ics=[]; breadth=[]; dates=[]
 for dt in x.index:
  a=x.loc[dt]; b=fwd.loc[dt]; z=a.notna()&b.notna()
  if z.sum()>=8:
   r=spearmanr(a[z],b[z]).statistic
   if np.isfinite(r): ics.append(r);breadth.append(z.sum());dates.append(dt)
 ar=np.array(ics); ic=ar.mean(); ir=ic/ar.std(ddof=1) if len(ar)>1 else np.nan
 hit=(ar>0).mean();
 # recent and two broad regimes
 print(f'h={h}: IC={ic:.6f} ICIR={ir:.6f} hit={hit:.4f} ICdates={len(ar)} meanN={np.mean(breadth):.3f}')
 for name,mask in [('2023-2026',pd.DatetimeIndex(dates)<pd.Timestamp('2027-01-01')),('2027+',pd.DatetimeIndex(dates)>=pd.Timestamp('2027-01-01')),('recent180',pd.DatetimeIndex(dates)>=x.index.max()-pd.Timedelta(days=180))]:
  q=ar[mask]
  if len(q)>1: print(f'  {name}: IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f} n={len(q)}')
# daily mean abs cross-sectional rank movement where eligible
r=x.rank(axis=1,pct=True); print('turnover mean daily rank abs-change',r.diff().abs().stack().mean())
