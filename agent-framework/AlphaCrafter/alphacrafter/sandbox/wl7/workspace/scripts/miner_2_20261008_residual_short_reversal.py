import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is None or len(d)<40: continue
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    c=pd.to_numeric(d['close'],errors='coerce'); r=c.pct_change()
    # residual short-term reversal: reverse last 3d return, but attenuate when 20d trend is strongly adverse
    r3=c/c.shift(3)-1; r20=c/c.shift(20)-1; vol=r.rolling(20).std()
    f=(-r3/(vol*np.sqrt(3))).where(vol>1e-8)
    # isolate short shock, avoiding simply buying persistent breakdowns
    f=f*(1-0.35*np.tanh(r20.abs()/vol.replace(0,np.nan)))
    z=pd.DataFrame({'f':f,'fr':c.shift(-1)/c-1,'r':r},index=c.index)
    z['s']=s; rows.append(z.reset_index())
x=pd.concat(rows,ignore_index=True)
ics=[]; turnovers=[]; nms=[]
for dt,g in x.groupby('date'):
    g=g.dropna(subset=['f','fr'])
    if len(g)>=8:
        ics.append(g['f'].corr(g['fr'],method='spearman'))
        nms.append(len(g))
# turnover as rank signal changes across dates per asset
p=x.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
turn=p.diff().abs().mean(axis=1).dropna().mean()
a=np.array(ics); print('dates',len(a),'avg_names',np.mean(nms),'coverage',sum(nms)/(len(nms)*15),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turnover',turn)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=x[(x.date>=lo)&(x.date<=hi)]; z=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=['f','fr'])
  if len(g)>=8:z.append(g.f.corr(g.fr,method='spearman'))
 z=np.array(z); print(lo,hi,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1) if len(z)>1 else np.nan)
for h in [5,10,20]:
 vals=[]
 for s,g in x.groupby('s'):
  c=(1+g.r.fillna(0)).cumprod() # not used
  # reconstruct forward h return from close unavailable; use cumulative daily return
  fr=(1+g.r.shift(-1)).rolling(h).apply(np.prod,raw=True).shift(-(h-1))-1
  gg=pd.DataFrame({'f':g.f,'fr':fr,'date':g.date}).dropna()
  for dt,dd in gg.groupby('date'):
   pass
 # cross-sectional h ret directly via return products by date/s
 y=x.copy(); y['fh']=y.groupby('s').r.shift(-1).rolling(h).apply(np.prod,raw=True).reset_index(level=0,drop=True).shift(-(h-1))
 # above alignment uncertain; omit
 print('horizon',h,'not computed')
