import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except Exception: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>100: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Trend persistence: medium return weighted by fraction of positive daily observations,
# with volatility normalization. All inputs are lagged before forward-return comparison.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
pos=(r>0).rolling(20,min_periods=15).mean()
fac=(p.pct_change(40)/vol)*(0.5+pos)
for h in [1,5,10,20]:
  ics=[]; cov=[]; turns=[]; recent=[]
  for i in range(1,len(p)-h):
    q=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
    if len(q)>=8:
      c=q.f.corr(q.r)
      if np.isfinite(c):
       ics.append(c); cov.append(len(q)/15)
       if i>=len(p)-250: recent.append(c)
    if i>1: turns.append((fac.iloc[i-1].rank()-fac.iloc[i-2].rank()).abs().mean()/15)
  a=np.asarray(ics); z=np.asarray(recent)
  print({'h':h,'dates':len(a),'avg_n':round(np.mean(cov)*15,2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),4),'coverage':round(np.mean(cov),4),'turnover':round(np.mean(turns),5),'recent250_IC':round(z.mean(),6),'recent250_ICIR':round(z.mean()/z.std(ddof=1),6)})
