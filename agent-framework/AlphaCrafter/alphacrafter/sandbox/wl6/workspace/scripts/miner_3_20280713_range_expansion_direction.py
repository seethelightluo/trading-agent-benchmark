import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s, days=4000)
            if d is not None: return d
        except (FileNotFoundError,KeyError,ValueError): pass
    return None
series={}
for s in U:
    d=fetch(s)
    if d is None or len(d)<100: continue
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
    r=d['close'].pct_change(); rng=(d['high']-d['low'])/d['close']
    expansion=(rng.rolling(3).mean()/rng.rolling(30).median()).clip(0.5,2.0)
    sig=r.rolling(5).sum()*expansion
    series[s]=pd.DataFrame({'sig':sig,'fwd1':d['close'].pct_change().shift(-1),'fwd5':d['close'].shift(-5)/d['close']-1,'fwd10':d['close'].shift(-10)/d['close']-1})
def eval(h):
    rows=[]
    for dt in sorted(set().union(*[x.index for x in series.values()])):
        a=[]
        for s,x in series.items():
            if dt in x.index and np.isfinite(x.loc[dt,'sig']) and np.isfinite(x.loc[dt,h]): a.append((x.loc[dt,'sig'],x.loc[dt,h],s))
        if len(a)>=8:
            z=pd.DataFrame(a,columns=['sig','ret','s']); rows.append((dt,z.sig.rank().corr(z.ret.rank()),len(a)))
    q=pd.DataFrame(rows,columns=['date','ic','n']); mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    turns=[]; prev=None
    for dt in q.date:
        valid=[s for s,x in series.items() if dt in x.index and np.isfinite(x.loc[dt,'sig'])]
        ranks={s:i for i,s in enumerate(sorted(valid,key=lambda s:series[s].loc[dt,'sig']))}
        if prev is not None:
            common=set(prev)&set(ranks)
            if common: turns.append(np.mean([abs(ranks[s]-prev[s])/(len(common)-1 or 1) for s in common]))
        prev=ranks
    print(h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(mean,6),'ICIR',round(icir,4),'hit',round((q.ic>0).mean(),4),'turn',round(np.mean(turns),5) if turns else None)
    for label,start,end in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026'),('2027-28','2027','2028')]:
        v=q[(q.date.astype(str)>=start)&(q.date.astype(str)<=end+'-12-31')].ic
        print(' ',label,len(v),round(v.mean(),6) if len(v) else None)
for h in ['fwd1','fwd5','fwd10']: eval(h)
print('assets',len(series))
