import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except Exception: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>100: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change();
# Multi-horizon alignment: volatility-normalized returns at 10/20/60d, weighted to medium trend,
# multiplied by agreement breadth. Score at t uses only data through t-1.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
z10=p.pct_change(10)/vol; z20=p.pct_change(20)/vol; z60=p.pct_change(60)/vol
align=((z10+2*z20+z60)/4)*(0.5+((np.sign(z10)+np.sign(z20)+np.sign(z60))>0).rolling(1).mean())
# agreement multiplier is 0.5 plus fraction of positive component horizons (cross-sectional per asset)
pos=((z10>0).astype(float)+(z20>0).astype(float)+(z60>0).astype(float))/3
fac=((z10+2*z20+z60)/4)*(0.5+pos)
for h in [1,5,10,20]:
  ics=[]; cov=[]; turn=[]; recent=[]
  for i in range(1,len(p)-h):
    q=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
    if len(q)>=8:
      c=q.f.corr(q.r)
      if np.isfinite(c):
       ics.append(c); cov.append(len(q)/15)
       if i>=len(p)-250: recent.append(c)
    if i>1: turn.append((fac.iloc[i-1].rank()-fac.iloc[i-2].rank()).abs().mean()/15)
  a=np.array(ics); rr=np.array(recent)
  print({'h':h,'dates':len(a),'avg_n':round(np.mean(cov)*15,2),'IC':round(a.mean(),5),'ICIR':round(a.mean()/a.std(ddof=1),5),'hit':round(np.mean(a>0),3),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turn),5),'recent250_IC':round(rr.mean(),5),'recent250_ICIR':round(rr.mean()/rr.std(ddof=1),5)})
