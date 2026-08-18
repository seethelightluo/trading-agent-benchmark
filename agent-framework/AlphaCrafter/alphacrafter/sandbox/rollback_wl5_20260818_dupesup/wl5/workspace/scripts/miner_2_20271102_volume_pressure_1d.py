import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# volume-confirmed pressure: recent signed intraday close location, scaled by abnormal volume
frames={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is None or len(d)<80: continue
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date')
    # CLV in [-1,1], signed close location; use volume when meaningful, otherwise equal weight
    rng=(d.high-d.low).replace(0,np.nan)
    clv=((2*d.close-d.high-d.low)/rng).clip(-1,1)
    vol=pd.to_numeric(d.volume,errors='coerce')
    vr=(vol/(vol.rolling(20,min_periods=10).median()+1e-12)).replace([np.inf,-np.inf],np.nan)
    # bounded volume confirmation, avoids crypto/index volume scale issues
    d['f']=(clv*vr.clip(0,3)).rolling(3,min_periods=3).mean()
    d['f_plain']=clv.rolling(3,min_periods=3).mean()
    d['ret1']=d.close.pct_change()
    for h in [1,5,10]: d[f'fwd{h}']=d.close.shift(-h)/d.close-1
    frames[s]=d.set_index('date')

for h in [1,5,10]:
    vals=[]; artifact=[]
    dates=sorted(set().union(*[set(x.index) for x in frames.values()]))
    for dt in dates:
        xs=[]; ys=[]
        for s,d in frames.items():
            if dt in d.index:
                row=d.loc[dt]
                if np.isscalar(row.f) and np.isfinite(row.f) and np.isfinite(row[f'fwd{h}']): xs.append(-row.f); ys.append(row[f'fwd{h}']); artifact.append((dt,s,-row.f))
        if len(xs)>=8: vals.append(pd.Series(xs).corr(pd.Series(ys),method='spearman'))
    a=pd.Series(vals).dropna(); mean=a.mean(); sd=a.std(ddof=1); icir=mean/sd if sd>0 else np.nan
    print('horizon',h,'dates',len(a),'mean_n',len(U),'IC',round(mean,6),'ICIR',round(icir,6),'hit',round((a>0).mean(),4))
    if h==1:
        out=pd.DataFrame(artifact,columns=['date','symbol','signal']); out.to_csv('scripts/miner_2_20271102_volume_pressure_1d_signal.csv',index=False)
# regime daily
h=1; rows=[]
for s,d in frames.items():
 for dt,r in d.iterrows():
  if np.isfinite(r.f) and np.isfinite(r.fwd1): rows.append((dt,s,-r.f,r.fwd1))
x=pd.DataFrame(rows,columns=['date','symbol','f','y']); x['date']=pd.to_datetime(x.date)
for label,mask in [('2020-22',(x.date<'2023-01-01')),('2023-24',((x.date>='2023-01-01')&(x.date<'2025-01-01'))),('2025-26',((x.date>='2025-01-01')&(x.date<'2027-01-01'))),('2027+',x.date>='2027-01-01')]:
 z=x[mask].groupby('date').apply(lambda q:q.f.corr(q.y,method='spearman')).dropna(); print('regime',label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None)
print('coverage_rows',len(x),'assets',x.symbol.nunique(),'last',x.date.max().date())
