import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        frames[s]=d
print('assets',len(frames), 'lengths', {k:len(v) for k,v in frames.items()})
# volume-confirmed breakout: 20d return, volatility adjusted, with volume participation confirmation
rows=[]
for s,d in frames.items():
    c=pd.to_numeric(d['close'],errors='coerce'); v=pd.to_numeric(d['volume'],errors='coerce')
    r=c.pct_change()
    # lag one day: all rolling values then shift(1)
    mom=c.pct_change(20)
    rv=r.rolling(20).std()*np.sqrt(252)
    vr=v.rolling(20).mean()/v.rolling(60).mean().replace(0,np.nan)
    # volume confirmation is bounded, neutral if absent
    vr=vr.replace([np.inf,-np.inf],np.nan).clip(.5,2.0)
    sig=(mom/(rv+0.01)) * (0.75+0.25*vr)
    sig=sig.shift(1)
    fwd=c.shift(-10)/c-1
    for dt in sig.index:
        if pd.notna(sig.loc[dt]) and pd.notna(fwd.loc[dt]): rows.append((dt,s,float(sig.loc[dt]),float(fwd.loc[dt])))
x=pd.DataFrame(rows,columns=['date','asset','factor','fwd'])
ics=[]
for dt,g in x.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ics.append((dt,g.factor.corr(g.fwd,method='spearman')))
ic=pd.Series(dict(ics)).dropna()
print('dates',len(ic),'avg instruments',x.groupby('date').size().mean(),'coverage',len(x)/sum(len(d) for d in frames.values()),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
# regime and decay horizons
for h in [1,5,10,20]:
    rr=[]
    for s,d in frames.items():
      c=pd.to_numeric(d.close,errors='coerce'); v=pd.to_numeric(d.volume,errors='coerce'); r=c.pct_change()
      sig=(c.pct_change(20)/(r.rolling(20).std()*np.sqrt(252)+.01))*(.75+.25*(v.rolling(20).mean()/v.rolling(60).mean()).clip(.5,2)).shift(1)
      z=pd.DataFrame({'f':sig,'y':c.shift(-h)/c-1}).dropna().reset_index(); z['asset']=s; rr.append(z)
    z=pd.concat(rr)
    a=[]
    for dt,g in z.groupby('date'):
      if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:a.append(g.f.corr(g.y,method='spearman'))
    a=pd.Series(a).dropna(); print('h',h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-27','2025','2027-12-31')]:
 q=ic[(ic.index>=lo)&(ic.index<=hi)]; print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('turnover approximate omitted')
