import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except FileNotFoundError: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>=120: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Medium-term return rewarded, but penalized by downside rather than total risk.
# Score is observable at t-1 and predicts t..t+h-1.
down=r.where(r<0,0.0).rolling(40,min_periods=25).std()*np.sqrt(40)
fac=p.pct_change(20)/down.replace(0,np.nan)
for h in [1,5,10,20]:
  ics=[]; cov=[]; turns=[]; recent=[]; nreg=[]
  for i in range(61,len(p)-h):
    z=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
    if len(z)>=8:
      q=z.f.corr(z.r)
      if np.isfinite(q):
        ics.append(q); cov.append(len(z)/15); nreg.append(len(z))
        if i>=len(p)-250: recent.append(q)
      if i>61:
        a=fac.iloc[i-1].rank(pct=True); b=fac.iloc[i-2].rank(pct=True)
        turns.append((a-b).abs().mean())
  a=np.array(ics); rr=np.array(recent)
  print({'h':h,'dates':len(a),'avg_n':round(np.mean(nreg),2),'IC':round(a.mean(),5),'ICIR':round(a.mean()/a.std(ddof=1),5),'hit':round(np.mean(a>0),3),'coverage':round(np.mean(cov),3),'turnover':round(np.mean(turns),5),'recent250_IC':round(rr.mean(),5),'recent250_ICIR':round(rr.mean()/rr.std(ddof=1),5)})
