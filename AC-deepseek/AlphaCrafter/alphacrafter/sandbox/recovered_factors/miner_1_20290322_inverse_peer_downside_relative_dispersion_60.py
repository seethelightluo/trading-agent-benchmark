"""One idea: inverse peer-downside relative dispersion (60 sessions).
On market-down (cross-asset median-return <0) sessions, measure each asset's
standard deviation of return in excess of the peer median. Assets with lower
idiosyncratic downside dispersion are hypothesized to have more resilient
subsequent cross-asset returns. The score is fully price-only and lagged."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C); R=P.pct_change(); peer=R.median(axis=1); rel=R.sub(peer,axis=0)
# Use only downside peer sessions, while retaining a fixed calendar 60-observation window.
down=rel.where(peer<0)
# require 20 peer-down sessions; score is inverse dispersion of excess performance.
f=(-down.rolling(60,min_periods=20).std()).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20]; cutoff=P.dropna(how='all').index.max(); FW={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=FW[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR inverse_peer_downside_relative_dispersion_60 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in H:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
