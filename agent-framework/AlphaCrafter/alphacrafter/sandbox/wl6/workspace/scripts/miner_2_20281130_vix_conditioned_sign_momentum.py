import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None:return d
        except Exception: pass
S={}
for s in U:
    d=fetch(s)
    if d is None: continue
    d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=d.close; r=c.pct_change()
    vol=r.rolling(40).std(); base=r.rolling(20).sum()/(vol*np.sqrt(20))
    persistence=(r.gt(0).rolling(20).mean()-0.5).abs()*2
    S[s]=pd.DataFrame({'sig':base*(0.5+0.5*persistence),'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
# Observation-only VIX, lagged one completed session. Stress is a rolling percentile.
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').sort_index().close
stress=v.rolling(252,min_periods=60).rank(pct=True).shift(1).clip(0,1)
for s,x in S.items():
    x['sig']=x.sig.reindex(x.index)*(1.0-0.40*stress.reindex(x.index))
def evaluate(col):
    out=[]
    for dt in sorted(set().union(*[x.index for x in S.values()])):
        a=[(x.loc[dt].sig,x.loc[dt][col]) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt][col])]
        if len(a)>=8:
            z=pd.DataFrame(a,columns=['s','r']); out.append((dt,z.s.rank().corr(z.r.rank()),len(a)))
    q=pd.DataFrame(out,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
    print(col,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.mean()/15,4),'assets',len(S))
    for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
        w=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic
        print('regime',a,len(w),round(w.mean(),6) if len(w) else None)
    return q
for col in ['f1','f5','f10']: evaluate(col)
