import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
        D[s]=x
# Candidate: volume-confirmed short-horizon shock reversal, with abnormal volume capped and volatility normalization.
rows=[]
for s,x in D.items():
    c=x['close'].astype(float); v=x['volume'].astype(float).replace(0,np.nan)
    r=c.pct_change()
    vol=r.rolling(20,min_periods=15).std()
    abv=(v/v.rolling(30,min_periods=20).median()).clip(0.5,3.0)
    # high-volume negative shocks mean-revert; normalize by trailing risk, all lagged one bar
    sig=(-(c.pct_change(3))/vol * (1+0.35*(abv-1))).shift(1)
    fwd=c.pct_change().shift(-1)
    for dt in sig.index:
        if pd.notna(sig.loc[dt]) and pd.notna(fwd.loc[dt]): rows.append((dt,s,float(sig.loc[dt]),float(fwd.loc[dt])))
z=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
ics=[]; nobs=[]
for dt,g in z.groupby('date'):
    g=g.dropna()
    if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
        ics.append((dt,g.factor.rank().corr(g.fwd.rank()))); nobs.append(len(g))
ics=pd.Series(dict(ics)).dropna(); ics.index=pd.to_datetime(ics.index)
print('dates',len(ics),'avg_n',np.mean(nobs),'coverage',len(z)/((z.date.max()-z.date.min()).days+1)/len(U))
print('IC %.8f ICIR %.8f hit %.4f turnover_n/a' % (ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean()))
# turnover rank ordering on common dates
p=z.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
turn=p.diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31')]:
 q=ics[(ics.index>=pd.Timestamp(lo))&(ics.index<=pd.Timestamp(hi))]; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [1,3,5,10]:
 rr=[]
 for s,x in D.items():
  c=x.close.astype(float); v=x.volume.astype(float).replace(0,np.nan); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std(); abv=(v/v.rolling(30,min_periods=20).median()).clip(.5,3); sig=(-(c.pct_change(3))/vol*(1+.35*(abv-1))).shift(1); fw=c.pct_change(h).shift(-h+1)
  for dt in sig.index:
   if pd.notna(sig.loc[dt]) and pd.notna(fw.loc[dt]): rr.append((dt,s,sig.loc[dt],fw.loc[dt]))
 q=pd.DataFrame(rr,columns=['date','s','f','r']); a=[]
 for _,g in q.groupby('date'):
  if len(g)>=8: a.append(g.f.rank().corr(g.r.rank()))
 a=pd.Series(a).dropna(); print('decay',h,len(a),a.mean())
