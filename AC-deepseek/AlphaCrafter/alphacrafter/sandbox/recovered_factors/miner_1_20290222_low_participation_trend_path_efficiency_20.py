"""One idea: low-participation trend-path efficiency, trailing 20 observations.
High score means an asset made an efficient directional move with below-peer trading
participation, hypothesizing that under-owned/low-crowding trends persist."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; P={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 P[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce')
p=pd.DataFrame(P); volu=pd.DataFrame(V); r=p.pct_change()
# Signed 20d endpoint move, divided by total path movement; reward efficiency only
# when log-volume is below the asset's own trailing typical participation.
endpoint=p.pct_change(20); efficiency=endpoint/(r.abs().rolling(20,min_periods=12).sum().replace(0,np.nan))
lv=np.log1p(volu); participation=(lv-lv.rolling(60,min_periods=20).mean())/lv.rolling(60,min_periods=20).std().replace(0,np.nan)
# Smooth inverse participation to prevent a single volume observation dominating.
invpart=(-participation.rolling(5,min_periods=3).mean()).clip(-3,3)
f=(efficiency*invpart).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.median(axis=1),axis=0)
cut=p.dropna(how='all').index.max(); hs=[1,5,10,20]; fw={h:p.shift(-h)/p-1 for h in hs}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(ic): z.append(ic);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.asarray(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR low_participation_trend_path_efficiency_20 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in hs: print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
