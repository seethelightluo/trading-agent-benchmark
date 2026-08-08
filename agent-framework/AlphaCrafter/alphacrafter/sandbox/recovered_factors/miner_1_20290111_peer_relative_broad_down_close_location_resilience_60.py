"""One idea: peer-relative close-location resilience on broad negative sessions, 60 observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={};H={};L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce');H[a]=pd.to_numeric(d.high,errors='coerce');L[a]=pd.to_numeric(d.low,errors='coerce')
p=pd.DataFrame(C).sort_index(); hi=pd.DataFrame(H).reindex(p.index);lo=pd.DataFrame(L).reindex(p.index)
r=p.pct_change(fill_method=None); m=r.median(axis=1)
clv=(p-lo)/(hi-lo).replace(0,np.nan)
# On broad down sessions, score each asset's closing strength relative to its contemporaneous peer median.
# Averaging only these relative residuals isolates defensive intraday demand from general market close-location.
res=clv.sub(clv.median(axis=1),axis=0).where(m<0)
f=res.rolling(60,min_periods=12).mean();f=f.sub(f.median(axis=1),axis=0)
fw={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; out=[]; breadth=[]
 for t in x.index:
  z=pd.concat([x.loc[t],fw[h].loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append(q);breadth.append(len(z))
 if not out:return dict(dates=0)
 s=np.array(out);return dict(dates=len(s),ic=round(float(s.mean()),6),icir=round(float(s.mean()/s.std(ddof=1)),6),hit=round(float((s>0).mean()),4),mean_n=round(float(np.mean(breadth)),2),min_n=int(min(breadth)))
cut=p.dropna(how='all').index.max()
print('FACTOR peer_relative_broad_down_close_location_resilience_60 CUTOFF',cut.date(),'PERIOD',p.index.min().date(),cut.date(),'ASSETS',len(A))
print('CELLS',int(f.notna().sum().sum()),'OF',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6),'MEAN_NAMES',round(float(f.notna().sum(axis=1).mean()),2))
for h in [1,5,10,20]:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
