import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

UNIVERSE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=4000) for s in UNIVERSE}
D={s:x.sort_values('date').drop_duplicates('date').set_index('date') for s,x in D.items() if x is not None and len(x)>100}
# Candidate: 40d directional momentum, weighted by path efficiency and volatility-normalized.
# efficiency=abs(net move)/sum(abs daily moves), so noisy trends are penalized.
def signal(x, i):
    c=x.close.iloc[:i+1].astype(float)
    if len(c)<61: return np.nan
    r40=c.iloc[-1]/c.iloc[-41]-1
    dr=c.pct_change().iloc[-40:].dropna()
    eff=abs(r40)/(dr.abs().sum()+1e-12)
    vol=dr.std(ddof=1)*np.sqrt(20)
    return r40*eff/(vol+0.01)
# aligned dates; use next 10 completed observations per asset
all_dates=sorted(set().union(*[set(x.index) for x in D.values()]))
rows=[]
for dt in all_dates:
    vals=[]; fw=[]
    for s,x in D.items():
        pos=x.index.searchsorted(dt)
        if pos>=len(x) or x.index[pos]!=dt: continue
        if pos+10>=len(x): continue
        z=signal(x,pos)
        if np.isfinite(z): vals.append(z); fw.append(x.close.iloc[pos+10]/x.close.iloc[pos]-1)
    if len(vals)>=8:
        ic=pd.Series(vals).corr(pd.Series(fw),method='spearman')
        rows.append((dt,ic,len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
# horizons are exactly 10 trading observations from each asset, valid cross-section dates
print('candidate=path_efficiency_momentum_40d; dates=%d avg_n=%.2f coverage_dates=%.4f assets=%d' %(len(r),r.n.mean(),len(r)/len(all_dates),len(D)))
print('IC=%.8f ICIR=%.8f hit=%.6f' %(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
print('regimes:')
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029YTD','2029-01-01','2029-07-11')]:
 q=r[(r.date>=a)&(r.date<=b)]
 print(name,len(q), 'ic=%.8f icir=%.5f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>2 else 'NA')
# decay using common 1/5/20 horizons
for h in [1,5,10,20]:
 out=[]
 for dt in all_dates:
  vals=[]; fw=[]
  for s,x in D.items():
   pos=x.index.searchsorted(dt)
   if pos>=len(x) or x.index[pos]!=dt or pos+h>=len(x): continue
   z=signal(x,pos)
   if np.isfinite(z): vals.append(z); fw.append(x.close.iloc[pos+h]/x.close.iloc[pos]-1)
  if len(vals)>=8: out.append(pd.Series(vals).corr(pd.Series(fw),method='spearman'))
 out=pd.Series(out).dropna(); print('decay_%dd ic=%.8f n=%d'%(h,out.mean(),len(out)))
