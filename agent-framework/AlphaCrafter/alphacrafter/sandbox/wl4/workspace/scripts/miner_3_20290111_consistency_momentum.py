import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=2600)
    except FileNotFoundError: x=get_stock_daily_data(s,days=2600)
    if x is not None and len(x)>=150: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Consistency-weighted risk-adjusted 20d momentum; all inputs are lagged at scoring time.
trend=p.pct_change(20); consistency=(r>0).rolling(20,min_periods=15).mean(); fac=(trend/vol)*(0.5+consistency)
for h in [1,5,10,20]:
    ics=[]; cov=[]; turns=[]; recent=[]
    for i in range(1,len(p)-h):
        z=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
        if len(z)>=8:
            q=z.f.corr(z.r)
            if np.isfinite(q):
                ics.append(q); cov.append(len(z)/15)
                if i>=len(p)-250: recent.append(q)
            if i>1: turns.append((fac.iloc[i-1].rank()-fac.iloc[i-2].rank()).abs().mean()/15)
    a=np.array(ics); print({'h':h,'dates':len(a),'avg_n':round(np.mean(cov)*15,2),'IC':round(a.mean(),5),'ICIR':round(a.mean()/a.std(ddof=1),5),'hit':round(np.mean(a>0),3),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turns),5),'recent250_IC':round(np.mean(recent),5),'recent250_ICIR':round(np.mean(recent)/np.std(recent,ddof=1),5)})
