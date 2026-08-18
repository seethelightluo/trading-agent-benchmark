import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except Exception: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>100: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); r5=p.pct_change(5); r20=p.pct_change(20)
# Reversal is strengthened when the broad cross-asset tape is one-sided.
breadth=(r20>0).sum(axis=1)/r20.notna().sum(axis=1)
one_sided=(breadth-0.5).abs()*2
fac=-r5.mul(0.5+one_sided,axis=0)
print('cutoff',p.index[-1].date(),'universe',len(D),'rows',len(p))
for h in [1,5,10,20]:
  ics=[]; recent=[]; cov=[]; turns=[]
  for i in range(1,len(p)-h):
    q=pd.concat([fac.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
    if len(q)>=8:
      c=q.f.corr(q.r)
      if np.isfinite(c):
        ics.append(c); cov.append(len(q)/len(D))
        if i>=len(p)-261: recent.append(c)
    if i>1:
      turns.append((fac.iloc[i-1].rank(pct=True)-fac.iloc[i-2].rank(pct=True)).abs().mean())
  a=np.array(ics); z=np.array(recent)
  print({'h':h,'dates':len(a),'avg_n':round(np.mean(cov)*len(D),2),'IC':round(a.mean(),6),'ICIR':round(a.mean()/a.std(ddof=1),6),'hit':round(np.mean(a>0),4),'coverage':round(np.mean(cov),4),'turnover':round(np.mean(turns),6),'recent261_IC':round(z.mean(),6),'recent261_ICIR':round(z.mean()/z.std(ddof=1),6)})
