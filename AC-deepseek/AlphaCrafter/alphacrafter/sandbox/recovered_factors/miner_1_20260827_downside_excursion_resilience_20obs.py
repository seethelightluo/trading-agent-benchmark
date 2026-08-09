"""Validate one idea: 20-observation downside-excursion resilience.
Signal is negative mean intraday downside excursion from prior close, scaled by range.
Higher means price has experienced shallower downside probes relative to its daily range.
"""
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; prices={}; sig={}; libs={}
for a in assets:
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); d=d.sort_index()
 p=pd.to_numeric(d.close,errors='coerce').dropna(); r=p.pct_change(fill_method=None); prices[a]=p
 h=pd.to_numeric(d.high,errors='coerce').reindex(p.index); l=pd.to_numeric(d.low,errors='coerce').reindex(p.index)
 # One-day downside probe from yesterday's close, normalized by that day's range.
 probe=((p.shift(1)-l)/(h-l).replace(0,np.nan)).clip(lower=0)
 sig[a]=-probe.rolling(20,min_periods=15).mean()
 libs.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=p.pct_change(20,fill_method=None)/r.rolling(20,min_periods=15).std()
 libs.setdefault('miner_1_ravmom_20obs',{})[a]=p.pct_change(20,fill_method=None)/r.rolling(20,min_periods=15).std()
 libs.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-p.pct_change(5,fill_method=None)/r.rolling(5,min_periods=4).std()
 libs.setdefault('miner_2_realized_volatility_20obs',{})[a]=-r.rolling(20,min_periods=15).std()
 libs.setdefault('miner_2_volscaled_reversal_1obs',{})[a]=-r/r.rolling(20,min_periods=15).std()
 v=pd.to_numeric(d.volume,errors='coerce').reindex(p.index)
 libs.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(v/v.rolling(20,min_periods=15).mean())
panel=pd.DataFrame(prices).sort_index(); fac=pd.DataFrame(sig).reindex(panel.index)
def rc(x,y):
 z=pd.concat([x,y],axis=1).dropna(); return np.nan if len(z)<8 else z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
def calc(h,detail=False):
 fw=panel.shift(-h)/panel-1; ic=pd.Series([rc(fac.iloc[i],fw.iloc[i]) for i in range(len(panel))],index=panel.index).dropna(); turns=[]
 for i in range(1,len(panel)):
  z=pd.concat([fac.iloc[i-1],fac.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('H',h,'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'dates',len(ic),'mean_names',round(fac.notna().sum(axis=1).mean(),2),'turnover',round(np.mean(turns),6))
 if detail:
  for n,a,b in [('2020','2020-01-01','2020-12-31'),('2021_2022','2021-01-01','2022-12-31'),('2023_2024','2023-01-01','2024-12-31'),('2025_2026_08','2025-01-01','2026-12-31')]:
   q=ic.loc[a:b]; print('REGIME',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',len(q))
for h in (1,5,10,20): calc(h,h==5)
print('coverage',round(fac.notna().mean().mean(),4),'range',panel.index.min().date(),panel.index.max().date(),'assets',len(assets))
for n,x in libs.items():
 z=pd.concat([fac.stack().rename('a'),pd.DataFrame(x).stack().rename('b')],axis=1).dropna(); print('LIBCORR',n,round(z.a.rank().corr(z.b.rank()),6),'cells',len(z))
