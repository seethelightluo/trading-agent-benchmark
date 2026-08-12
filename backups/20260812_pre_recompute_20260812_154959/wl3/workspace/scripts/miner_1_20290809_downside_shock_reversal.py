import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        raw[s]=d['close'].astype(float)
p=pd.DataFrame(raw).sort_index().ffill(limit=3)
r=np.log(p).diff()
# Candidate: downside-adjusted 3d shock reversal, robust to cross-asset scale.
down=r.clip(upper=0).rolling(30,min_periods=15).std()
shock=r.rolling(3,min_periods=3).sum()
f=(-shock/(down*np.sqrt(3))).shift(1)
# cross sectional demean; avoid missing
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,3,5,10]:
    fr=np.log(p.shift(-h)/p)
    vals=[]; dates=[]; ns=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8:
            vals.append(x[ok].corr(y[ok],method='spearman')); dates.append(dt); ns.append(ok.sum())
    z=pd.Series(vals,index=dates).dropna()
    print('H',h,'dates',len(z),'avg_n',round(float(np.mean(ns)),2),'IC',round(float(z.mean()),7),'ICIR',round(float(z.mean()/z.std(ddof=1)),7),'hit',round(float((z>0).mean()),4))
# regime splits 1d
fr=np.log(p.shift(-1)/p); vals=[]; dates=[]
for dt in f.index:
    ok=f.loc[dt].notna()&fr.loc[dt].notna()
    if ok.sum()>=8:
        vals.append(f.loc[dt,ok].corr(fr.loc[dt,ok],method='spearman')); dates.append(dt)
z=pd.Series(vals,index=pd.to_datetime(dates)).dropna()
print('REGIMES')
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2029-12-31')]:
 q=z.loc[a:b]; print(a,b,len(q),round(q.mean(),7),round(q.mean()/q.std(ddof=1),7) if len(q)>1 else None)
print('coverage',round(float(f.notna().sum().sum()/f.size),4))
print('turnover',round(float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)).mean()),4))
# save signal artifact
out=f.copy(); out.to_csv('scripts/miner_1_20290809_downside_shock_reversal_signal.csv')
