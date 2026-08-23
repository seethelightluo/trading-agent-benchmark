import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    f=os.path.join(base,s+'.csv')
    d=pd.read_csv(f)
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').set_index('date')
    px[s]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index()
# Candidate: acceleration of medium-term trend, volatility normalized.
r=prices.pct_change()
ret20=prices/prices.shift(20)-1
ret60=prices/prices.shift(60)-1
vol60=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(ret20-ret60/3)/(vol60+0.02)
ics=[]; turnovers=[]; counts=[]
for i in range(len(prices)-10):
    dt=prices.index[i]; nxt=prices.iloc[i+1:i+11].iloc[-1]/prices.iloc[i]-1
    x=f.iloc[i]
    ok=x.notna()&nxt.notna()
    if ok.sum()>=8:
        ics.append(spearmanr(x[ok],nxt[ok]).statistic); counts.append(ok.sum())
        turnovers.append((x[ok].rank(pct=True)-f.iloc[i-1][ok].rank(pct=True)).abs().mean() if i else np.nan)
a=np.array(ics); a=a[np.isfinite(a)]
print(json.dumps({'factor':'volatility_normalized_trend_acceleration','dates':len(a),'avg_instruments':float(np.mean(counts)),'coverage':float(np.mean(counts)/15),'IC':float(a.mean()),'ICIR':float(a.mean()/a.std(ddof=1)*np.sqrt(len(a))),'hit_ratio':float(np.mean(a>0)),'turnover_proxy':float(np.nanmean(turnovers))},indent=2))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 z=[]
 for dt,v in zip(prices.index[0:len(ics)],ics):
  if str(dt)[:4]>=lo and str(dt)[:10]<=hi:z.append(v)
 print(lo, float(np.mean(z)) if z else None, len(z))
