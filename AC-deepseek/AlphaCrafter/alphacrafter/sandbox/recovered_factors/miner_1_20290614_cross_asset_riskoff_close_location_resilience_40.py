"""One idea: cross-asset risk-off close-location resilience (40 sessions)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']; frames={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    frames[a]=d.sort_values('date').set_index('date')
def panel(col): return pd.DataFrame({a:pd.to_numeric(frames[a][col],errors='coerce') for a in assets}).sort_index()
p=panel('close'); hi=panel('high'); lo=panel('low')
r=p.pct_change(); peer=r.median(axis=1)
# On broad risk-off sessions (median return below zero), a high close location signals absorption.
loc=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
riskoff=peer<0
f=loc.where(riskoff, np.nan).rolling(40,min_periods=12).mean().shift(1)
f=f.sub(f.median(axis=1),axis=0)
fw={h:p.shift(-h)/p-1 for h in (1,5,10,20)}; cutoff=p.dropna(how='all').index.max()
def ev(h, span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; nn=[]
 for dt in x.index:
  q=pd.concat((x.loc[dt],y.loc[dt]),axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR cross_asset_riskoff_close_location_resilience_40 cutoff',cutoff.date(),'assets',len(assets))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2),'riskoff_freq',round(float(riskoff.mean()),6))
for h in fw: print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
