import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-06-13')
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
rows=[]
for s,d in D.items():
    c=d.close.astype(float)
    r=c.pct_change()
    # Medium-horizon momentum, scaled by recent risk and gated by the slower trend.
    # The positive-part gate avoids treating a falling long trend as ordinary momentum.
    mom10=c.pct_change(10)
    vol20=r.rolling(20,min_periods=12).std()*np.sqrt(10)
    trend60=c.pct_change(60)
    gate=np.where(trend60>=0,1.0,-0.35)
    f=(mom10/(vol20+1e-12)*gate).shift(1)
    rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
P=pd.concat(rows,ignore_index=True)
def eval_h(h, subset=None):
    rr=[]
    for s,d in D.items():
        c=d.close.astype(float); fr=c.shift(-h)/c-1
        z=P[P.asset==s].set_index('date').f.reindex(c.index)
        rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':z.values,'fr':fr.values}))
    x=pd.concat(rr,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
    if subset: x=x[x.date.dt.year.between(*subset)]
    vals=[]; ns=[]
    for dt,g in x.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
            vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
    z=pd.Series(vals)
    return dict(dates=len(z),avg_n=round(float(np.mean(ns)),2),ic=round(float(z.mean()),8),icir=round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),5),hit=round(float((z>0).mean()),5),coverage=round(len(x)/(x.date.nunique()*len(U)),5))
print('assets',len(D),'dates',P.date.nunique())
for h in [1,5,10,20]: print('horizon',h,eval_h(h))
for reg in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',reg,eval_h(1,reg))
r=P.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',round(float(r.diff().abs().mean().mean()),5))
P.to_csv('scripts/miner_2_20270614_gated_momentum_signal.csv',index=False)
