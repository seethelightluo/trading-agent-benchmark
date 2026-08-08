"""One idea: asset-specific downside close-location reversal (40 sessions)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}; H={}; L={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 P[a]=pd.to_numeric(d.close,errors='coerce');H[a]=pd.to_numeric(d.high,errors='coerce');L[a]=pd.to_numeric(d.low,errors='coerce')
P=pd.DataFrame(P).sort_index(); H=pd.DataFrame(H).reindex(P.index); L=pd.DataFrame(L).reindex(P.index); R=P.pct_change()
# On each asset's own negative-return days, low close location captures unresolved selling;
# negate it: unusually weak closes are tested as a medium-horizon reversal signal.
loc=(P-L).div((H-L).replace(0,np.nan)); down=R.lt(0)
count=down.astype(float).rolling(40,min_periods=25).sum()
f=(-loc.where(down)).rolling(40,min_periods=25).mean().where(count>=12).shift(1)
f=f.sub(f.median(axis=1),axis=0)
FW={h:P.shift(-h).div(P)-1 for h in [1,5,10,20]};cutoff=P.dropna(how='all').index.max()
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=FW[h].reindex(x.index);z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':min(ns)}
print('FACTOR asset_downside_close_location_reversal_40 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2),'down_share',round(float(down.stack().mean()),4))
for h in [1,5,10,20]:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
