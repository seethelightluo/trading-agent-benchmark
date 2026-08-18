import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in A:
    try:
        d=get_stock_daily_data(a,days=1800)
        if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
    except Exception as e: print('ERR',a,e)
p=pd.concat(px,axis=1).sort_index().ffill(); r=px and px and px # noop
rets=p.pct_change()
print('assets',len(px),list(px),'range',p.index.min(),p.index.max())
# factor: risk-adjusted trend efficiency = signed 60d return / sum abs daily returns, captures persistent trend
factors={
 'trend_efficiency_60': p.pct_change(60)/(rets.abs().rolling(60).sum()),
 'momentum_20':p.pct_change(20),
 'reversal_5':-p.pct_change(5),
 'invvol_20':-rets.rolling(20).std(),
}
for name,f in factors.items():
  obs=[]; turnovers=[]; cov=[]
  for i in range(len(p)-1):
    date=p.index[i]; nxt=rets.iloc[i+1]
    z=f.iloc[i]
    q=pd.concat([z.rename('f'),nxt.rename('y')],axis=1).dropna()
    if len(q)>=8:
      obs.append(q.f.corr(q.y)); cov.append(len(q)/15)
    if i>0:
      prev=f.iloc[i-1]; both=pd.concat([z,prev],axis=1).dropna()
      if len(both)>=8: turnovers.append((both.iloc[:,0].rank()!=both.iloc[:,1].rank()).mean())
  x=np.array(obs); print(name,'dates',len(x),'meanIC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'coverage',np.mean(cov),'turn',np.mean(turnovers))
  for h in [1,5,10]:
    oo=[]
    for i in range(len(p)-h):
      q=pd.concat([f.iloc[i].rename('f'),(p.pct_change(h).iloc[i+h]).rename('y')],axis=1).dropna()
      if len(q)>=8: oo.append(q.f.corr(q.y))
    print(' decay',h,round(float(np.nanmean(oo)),5),len(oo))
