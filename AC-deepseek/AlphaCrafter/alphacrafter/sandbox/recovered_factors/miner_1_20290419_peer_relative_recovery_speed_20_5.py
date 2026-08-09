"""One idea: peer-relative recovery speed after ordinary weakness (20d/5d).
The score compares a recent five-session peer-relative recovery with the
asset's preceding 20-session peer-relative downside semideviation. Unlike a
drawdown gate, every asset with sufficient history is ranked, testing whether
recovery per unit of prior relative downside has cross-asset persistence.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C); r=P.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
# Latest relative recovery, scaled only by prior (excluding latest block) relative downside.
recent=rel.rolling(5,min_periods=4).sum()
prior=rel.shift(5)
down=np.sqrt((prior.clip(upper=0).pow(2)).rolling(20,min_periods=15).mean())
f=(recent/(down+0.0005)).shift(1)
f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20]; cutoff=P.dropna(how='all').index.max(); FW={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=FW[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_recovery_speed_20_5 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in H: print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
