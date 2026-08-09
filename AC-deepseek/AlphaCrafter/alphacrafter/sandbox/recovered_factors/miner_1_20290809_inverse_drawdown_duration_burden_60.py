"""One idea: inverse drawdown-duration burden (60).
Tests whether assets in deeper and more persistent own drawdowns subsequently lag;
negative burden favours shallow/recent drawdowns without using forward data.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); R=P.pct_change(); peak=P.rolling(60,min_periods=45).max(); dd=P.div(peak)-1
# Age since latest 60d high: 0 at a high, rising while underwater (strictly trailing information).
at_high=P.ge(peak*(1-1e-12)); age=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 g=(at_high[a].fillna(False)).cumsum(); age[a]=at_high[a].groupby(g).cumcount().where(P[a].notna())
# Negative sign means high values select less deeply and less persistently impaired assets.
burden=(-dd)*age/60
f=(-burden).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20]; FW={h:P.shift(-h).div(P)-1 for h in H}; cutoff=P.dropna(how='all').index.max()
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=FW[h].reindex(x.index); z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR inverse_drawdown_duration_burden_60 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
print('MEAN_DD',round(float(dd.mean().mean()),6),'MEAN_AGE',round(float(age.mean().mean()),3),'MEAN_BURDEN',round(float(burden.mean().mean()),6))
for h in H:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
