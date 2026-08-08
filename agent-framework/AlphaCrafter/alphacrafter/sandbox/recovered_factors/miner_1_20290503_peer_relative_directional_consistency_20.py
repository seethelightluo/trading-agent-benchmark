"""One idea: peer-relative directional consistency (20 sessions).
The score is the signed efficiency of an asset's peer-relative return path:
20-day cumulative relative return divided by total absolute relative movement.
Unlike raw momentum it rewards a persistent relative path and discounts
oscillating moves. Inputs are lagged one session before forward-return tests.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; close={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
    close[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
p=pd.DataFrame(close); r=p.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
num=rel.rolling(20,min_periods=16).sum(); den=rel.abs().rolling(20,min_periods=16).sum().replace(0,np.nan)
f=(num/den).shift(1); f=f.sub(f.median(axis=1),axis=0)
fw={h:p.shift(-h).div(p).sub(1) for h in [1,5,10,20]}; cutoff=p.dropna(how='all').index.max()
def evaluate(h, span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); ics=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): ics.append(z); ns.append(len(q))
 if not ics:return {'dates':0}
 z=np.array(ics); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_directional_consistency_20 cutoff',cutoff.date(),'assets',len(assets))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in [1,5,10,20]:print('H',h,evaluate(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,evaluate(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
