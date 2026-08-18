import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=2600)
    except Exception: x=get_stock_daily_data(s,days=2600)
    if x is not None and len(x)>=150: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# One interpretable idea: volatility-normalized 10d trend, with cross-asset breadth gating.
trend=p.pct_change(10); vol=r.rolling(20,min_periods=15).std()*np.sqrt(10)
breadth=(r.rolling(5,min_periods=5).mean()>0).mean(axis=1)
# broad positive breadth permits trend-following; weak breadth reverses the trend signal
fac=trend/vol * np.where(breadth>=0.50,1.0,-1.0).reshape(-1,1)
# use prior completed date factor and non-overlapping forward returns for conservative date count
for h in [1,5,10,20]:
    ics=[]; ns=[]; turns=[]
    for i in range(1,len(p)-h):
        z=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
        if len(z)>=8:
            q=z.f.corr(z.r)
            if np.isfinite(q): ics.append(q); ns.append(len(z))
        if i>1:
            a=pd.Series(fac.iloc[i-1],index=p.columns).rank(); b=pd.Series(fac.iloc[i-2],index=p.columns).rank()
            turns.append((a-b).abs().mean()/15)
    a=np.asarray(ics)
    print({'h':h,'dates':len(a),'avg_n':round(float(np.mean(ns)),2),'IC':round(float(a.mean()),6),'ICIR':round(float(a.mean()/a.std(ddof=1)),6),'hit':round(float(np.mean(a>0)),4),'coverage':round(float(np.mean(ns)/15),4),'turnover':round(float(np.mean(turns)),6)})
print('date_range',p.index.min(),p.index.max(),'instruments',len(D))
