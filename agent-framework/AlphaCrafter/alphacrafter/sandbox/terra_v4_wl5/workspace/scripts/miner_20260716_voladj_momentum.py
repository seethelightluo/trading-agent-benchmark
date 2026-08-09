import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']
print('assets',len(assets),assets)
px={}
for a in assets:
    try:
        d=get_stock_daily_data(a,days=2000)
        if d is not None and len(d)>120: px[a]=d.set_index('date')['close'].astype(float)
    except Exception as e: print('err',a,e)
p=pd.concat(px,axis=1).sort_index()
# Factor: 60d trend normalized by trailing 20d realized volatility, ranked cross-section; forward 1d IC
ret=p.pct_change(); mom=p/p.shift(60)-1; vol=ret.rolling(20).std(); fac=mom/vol
ics=[]; turnovers=[]; used=0; nobs=[]
prev=None
for i in range(60,len(p)-1):
    f=fac.iloc[i]; y=ret.iloc[i+1]
    z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.f.corr(z.y)); nobs.append(len(z)); used+=1
        r=f.rank(pct=True)
        if prev is not None: turnovers.append((r-prev).abs().mean())
        prev=r
x=np.array(ics,dtype=float); x=x[np.isfinite(x)]
print('dates',len(p),'instruments',len(px),'IC_obs',len(x),'meanIC',x.mean(),'std',x.std(ddof=1),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'coverage_mean',np.mean(nobs)/len(assets),'turnover',np.mean(turnovers))
for h in [1,5,10]:
 ys=p.pct_change(h).shift(-h)
 q=[]
 for i in range(60,len(p)-h):
  z=pd.concat([fac.iloc[i].rename('f'),ys.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(z.f.corr(z.y))
 q=np.array(q);q=q[np.isfinite(q)]
 print('decay',h,len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1))
