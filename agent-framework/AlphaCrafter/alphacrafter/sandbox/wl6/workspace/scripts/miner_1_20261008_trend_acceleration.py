import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(s, days=2600)
            if x is not None and len(x): return x
        except Exception: pass
    return None
D={s:get(s) for s in U}
# common date frame, completed daily closes; acceleration is recent 10d return less prior 10d return, volatility scaled
rows=[]
for s,x in D.items():
    if x is None: continue
    x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date')
    c=pd.to_numeric(x['close'],errors='coerce'); r=np.log(c).diff()
    fac=(np.log(c/c.shift(10))-np.log(c.shift(10)/c.shift(20))) / r.rolling(20).std().shift(1)
    # signal at t, forward next-day log return
    z=pd.DataFrame({'date':x.date,'f':fac,'fr':np.log(c.shift(-1)/c)})
    z['asset']=s; rows.append(z)
a=pd.concat(rows).dropna()
ics=[]; turnovers=[]
for dt,g in a.groupby('date'):
    if len(g)>=8: ics.append(g['f'].corr(g['fr'],method='spearman'))
# rank turnover using consecutive available cross sections
ranks=a.assign(rank=a.groupby('date')['f'].rank(pct=True)).pivot(index='date',columns='asset',values='rank').sort_index()
turn=(ranks.diff().abs().mean(axis=1)).dropna().mean()
v=a.groupby('date').size()
ic=np.array(ics); print('dates',len(ic),'assets/date',v.mean(),'coverage',len(a)/(len(D)*max(v.index.nunique(),1)),'daily_ic',np.nanmean(ic),'icir',np.nanmean(ic)/np.nanstd(ic,ddof=1),'hit',np.mean(ic>0),'turnover',turn)
for h in [5,10]:
    z=[]
    for s,x in D.items():
      if x is None: continue
      c=pd.to_numeric(x['close'],errors='coerce'); r=np.log(c.shift(-h)/c)
      f=(np.log(c/c.shift(10))-np.log(c.shift(10)/c.shift(20)))/np.log(c).diff().rolling(20).std().shift(1)
      q=pd.DataFrame({'date':pd.to_datetime(x.date),'f':f,'r':r}).dropna(); z.append(q)
    q=pd.concat(z); ii=[g.f.corr(g.r,method='spearman') for _,g in q.groupby('date') if len(g)>=8]; ii=np.array(ii)
    print('horizon',h,'dates',len(ii),'ic',np.nanmean(ii),'icir',np.nanmean(ii)/np.nanstd(ii,ddof=1))
