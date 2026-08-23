import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-10')
def fetch(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,3000)
            if d is not None and len(d):
                d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize()
                return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
        except Exception: pass
    return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def make(h):
    out=[]
    for s,d in D.items():
        c=d.close.astype(float); r=c.pct_change(); net=c.pct_change(20)
        efficiency=net.abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)
        f=(net*efficiency).shift(1)
        out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.to_numpy(),'fr':(c.shift(-h)/c-1).to_numpy()}))
    return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(q):
    vals=[]; ns=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
            vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
    x=pd.Series(vals)
    return {'dates':len(x),'avg_n':round(float(np.mean(ns)),2),'ic':round(float(x.mean()),5),'icir':round(float(x.mean()/x.std(ddof=1)*np.sqrt(252)),4),'hit':round(float((x>0).mean()),4)}
print('assets',len(D),'range',min(x.index.min() for x in D.values()),max(x.index.max() for x in D.values()))
q1=make(1); print('raw_rows',len(q1),'dates',q1.date.nunique(),'coverage',round(q1.f.notna().mean(),4))
for h in [1,2,5,10,20]: print('horizon',h,stats(make(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,stats(q1[(q1.date.dt.year>=lo)&(q1.date.dt.year<=hi)]))
r=q1.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',round(float(r.diff().abs().mean().mean()),5))
q1.to_csv('scripts/miner_3_20270311_range_efficiency_signal.csv',index=False)
