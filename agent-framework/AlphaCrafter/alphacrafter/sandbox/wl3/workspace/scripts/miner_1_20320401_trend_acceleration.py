import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: x=None
        if x is not None and len(x): break
    if x is not None and len(x):
        D[s]=x.assign(date=pd.to_datetime(x.date)).sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=np.log(pd.DataFrame(D).sort_index().ffill()); r=p.diff()
# Trend acceleration: recent 20-day return versus average 20-day return over prior 40 days.
# Lagged one session, evaluated at 10-day forward return.
f=(r.rolling(20).sum()-r.shift(20).rolling(40).sum()/2).shift(1)
fr=p.shift(-10)-p
rows=[]
for dt in f.index:
    a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
turn=f.rank(pct=True).diff().abs().mean(axis=1).mean()
print('dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',turn)
for n in [60,120,252]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2032')]:
 x=q.loc[a:b]; print(a+'-'+b,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
f.to_csv('scripts/miner_1_20320401_trend_acceleration_signal.csv'); z.to_csv('scripts/miner_1_20320401_trend_acceleration_ic.csv')
