import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-10-08')
frames={}
for s in U:
    d=get_stock_daily_data(s,3000)
    if d is None or len(d)==0: d=get_index_daily_data(s,3000)
    if d is not None:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cut].sort_values('date')
        d['r']=d.close.pct_change(); frames[s]=d.set_index('date')

# Candidate: asymmetric trend efficiency. Positive-return share confirms trend,
# while downside volatility penalizes unstable/downside paths.
rows=[]
for s,d in frames.items():
    r=d.r
    ret10=d.close.pct_change(10)
    down=r.where(r<0,0).rolling(20).std()
    upfrac=(r>0).rolling(20).mean()
    # signal available after completed date t
    f=(ret10 * (0.5+upfrac) / (down*np.sqrt(10))).replace([np.inf,-np.inf],np.nan)
    fr=d.close.pct_change().shift(-1)
    for dt in d.index:
        if dt<=cut and pd.notna(f.get(dt)) and pd.notna(fr.get(dt)):
            rows.append((dt,s,f.loc[dt],fr.loc[dt]))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
ics=x.groupby('date').apply(lambda z: z.factor.corr(z.fwd),include_groups=False).dropna()
print('candidate asymmetric_trend_efficiency')
print('dates',len(ics),'avg_names',x.groupby('date').size().mean(),'coverage',x.groupby('date').size().mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(ics.mean(),ics.mean()/ics.std(),(ics>0).mean(),x.sort_values(['symbol','date']).groupby('symbol').factor.apply(lambda z:(z.rank(pct=True).diff().abs().mean())).mean()))
for h in [5,10,20]:
    q=[]
    for s,d in frames.items():
      rr=d.close.pct_change(h).shift(-h)
      ff=(d.close.pct_change(10)*(0.5+(d.r>0).rolling(20).mean())/(d.r.where(d.r<0,0).rolling(20).std()*np.sqrt(10))).replace([np.inf,-np.inf],np.nan)
      z=pd.DataFrame({'f':ff,'r':rr}).loc[:cut].dropna()
      q += [(dt, z.loc[dt,'f'],z.loc[dt,'r']) for dt in z.index]
    a=pd.DataFrame(q,columns=['date','f','r']).groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna()
    print('horizon',h,'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std()))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 a=ics.loc[lo:hi]; print('regime',lo,hi,'n',len(a),'ICIR',a.mean()/a.std() if len(a)>1 else np.nan,'IC',a.mean())
