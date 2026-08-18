import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except Exception: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>120: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Novel interpretable signal: 60d momentum scaled by volatility, with a mild
# volatility-state gate: reward momentum more when current vol is below its
# own 120d median (trend persistence), and fade it in high-vol state.
vol=r.rolling(20,min_periods=15).std()
state=(vol/(vol.rolling(120,min_periods=60).median())).clip(0.5,2.0)
fac=(p.pct_change(60)/vol)* (1.5-state).clip(-1,1)
for h in [1,5,10,20]:
    ics=[]; ns=[]; cov=[]; turns=[]; recent=[]
    for i in range(181,len(p)-h):
        z=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
        if len(z)>=8:
            q=z.f.corr(z.r)
            if np.isfinite(q):
                ics.append(q); ns.append(len(z)); cov.append(len(z)/15)
                if i>=len(p)-250: recent.append(q)
        if i>181:
            turns.append((fac.iloc[i-1].rank(pct=True)-fac.iloc[i-2].rank(pct=True)).abs().mean())
    a=np.array(ics); rr=np.array(recent)
    print({'h':h,'dates':len(a),'avg_n':round(np.mean(ns),2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),3),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turns),5),'recent250_IC':round(rr.mean(),6),'recent250_ICIR':round(rr.mean()/rr.std(ddof=1),6)})
