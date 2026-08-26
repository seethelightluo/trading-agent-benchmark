import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# acceleration: recent 20d return relative to prior 40d return, normalized by recent vol
xs={}
for s in U:
    d=get_index_daily_data(s, days=5000)
    if d is None or len(d)<150: continue
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    px=pd.to_numeric(d['close'],errors='coerce')
    r=np.log(px).diff()
    f=(np.log(px/px.shift(20))-np.log(px.shift(20)/px.shift(60))) / (r.rolling(60).std()*np.sqrt(20)+1e-12)
    # signal available at t, forward 10d return
    fr=np.log(px.shift(-10)/px)
    xs[s]=pd.DataFrame({'f':f,'fr':fr})
all_dates=sorted(set().union(*[set(x.index) for x in xs.values()]))
ics=[]; rows=[]
for dt in all_dates:
    a=[]
    for s,x in xs.items():
        if dt in x.index and np.isfinite(x.loc[dt,'f']) and np.isfinite(x.loc[dt,'fr']): a.append((s,x.loc[dt,'f'],x.loc[dt,'fr']))
    if len(a)>=8:
        z=pd.DataFrame(a,columns=['s','f','fr'])
        ic=z.f.corr(z.fr,method='spearman')
        if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(a)))
ser=pd.Series([x[1] for x in rows], index=[x[0] for x in rows])
print('dates',len(rows),'mean_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC',ser.mean(),'ICIR',ser.mean()/ser.std(ddof=1),'hit',np.mean(ser>0),'turnover_proxy',np.mean(np.abs(ser.diff())))
for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_27','2025-01-01','2027-12-31'),('2028_29','2028-01-01','2029-12-31'),('2030_now','2030-01-01','2033-05-20')]:
 q=ser[(ser.index>=lo)&(ser.index<=hi)]; print(name,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,np.mean(q>0) if len(q) else np.nan)
# horizon decay
for h in [5,10,20,40]:
 vals=[]
 for s,x in xs.items():
  pass
 # recompute using same f and prices unavailable; just report primary
