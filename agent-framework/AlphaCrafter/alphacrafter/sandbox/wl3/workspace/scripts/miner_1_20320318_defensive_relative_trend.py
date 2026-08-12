import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: x=fn(s,days=5000)
        except Exception: x=None
        if x is not None and len(x): break
    if x is not None and len(x):
        x=x.copy(); x.date=pd.to_datetime(x.date)
        D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); lp=np.log(p); r=lp.diff()
# Defensive-relative trend: each asset's 20d return relative to the contemporaneous
# defensive basket (gold and two synthetic 10y yield series), then cross-sectional demean.
defs=[x for x in ['XAU','US10Y','CN10Y'] if x in lp]
defret=lp[defs].diff(20).mean(axis=1)
f=lp.diff(20).sub(defret,axis=0); f=f.sub(f.mean(axis=1),axis=0).shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
    a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('shape',p.shape,'valid_dates',len(z),'assets',len(D),'avgN',z.n.mean(),'coverage',z.n.mean()/len(D))
print('H10 IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2032')]:
 x=q.loc[lo:hi]; print(lo,len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
for n in [60,120,252]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
f.to_csv('scripts/miner_1_20320318_defensive_relative_trend_signal.csv'); z.to_csv('scripts/miner_1_20320318_defensive_relative_trend_ic.csv')
