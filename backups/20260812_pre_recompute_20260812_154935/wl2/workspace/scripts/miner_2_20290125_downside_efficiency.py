import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); frames[s]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.DataFrame({s:d.close for s,d in frames.items()}); r=np.log(close).diff()
# One interpretable candidate: medium-term return divided by downside volatility,
# with a slow trend-consistency gate; all inputs lagged one completed bar.
down=np.sqrt((r.clip(upper=0)**2).rolling(60).mean())*np.sqrt(252)
trend=r.rolling(40).sum(); signal=(r.rolling(30).sum()/(down+1e-8))*(0.5+0.5*(r.rolling(60).sum()>0))
f=signal.shift(1)
for h in [1,5,10,20]:
 y=r.shift(-h).rolling(h).sum() if h>1 else r.shift(-1)
 vals=[]; dates=[]; ns=[]; rankturn=[]; prev=None
 for dt in f.index:
  x,z=f.loc[dt],y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],z[ok]).statistic
   if np.isfinite(q):
    vals.append(q); dates.append(dt); ns.append(int(ok.sum()))
    rr=x[ok].rank(pct=True)
    rankturn.append(np.mean(np.abs(rr-(prev if prev is not None else rr))))
    prev=rr
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(rankturn))
 # recent regime diagnostics
 for start in ['2020-01-01','2023-01-01','2026-01-01','2027-01-01']:
  b=a[np.array(dates)>=pd.Timestamp(start)]
  if len(b): print(start,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
print('assets',len(frames),'range',close.index.min(),close.index.max(),'coverage',close.notna().mean().mean())
# save signal artifact for provenance
out=f.copy(); out.to_csv('scripts/miner_2_20290125_downside_efficiency_signal.csv')
