import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    try: d=get_index_daily_data(s,3000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s,3000)
        except Exception: d=None
    if d is None or len(d)==0:return None
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Range-weighted intraday reversal: reversal is stronger when the session has a
# wide, informative range; normalize by rolling 20d median range to avoid scale bias.
rows=[]
for s,d in D.items():
    x=d[['open','close','high','low']].replace([np.inf,-np.inf],np.nan).dropna()
    intr=x.close/x.open-1
    rng=(x.high-x.low)/x.open
    scale=rng.rolling(20,min_periods=10).median()
    f=-(intr)*(rng/scale).clip(0,4)
    fr=x.close.shift(-1)/x.close-1
    z=pd.DataFrame({'f':f,'fr':fr,'asset':s}).dropna().reset_index(); rows.append(z)
R=pd.concat(rows,ignore_index=True); obs=[]
for dt,g in R.groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: obs.append((dt,g.f.corr(g.fr,method='spearman'),len(g)))
I=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date');
print('assets',len(D),'dates',len(I),'avg_n',I.n.mean()); print('daily',I.ic.mean(),I.ic.std(ddof=1),I.ic.mean()/I.ic.std(ddof=1)*np.sqrt(252),(I.ic>0).mean())
for h in [1,5,10,20]:
    rr=[]
    for s,d in D.items():
        x=d[['open','close','high','low']].replace([np.inf,-np.inf],np.nan).dropna(); intr=x.close/x.open-1; rng=(x.high-x.low)/x.open; scale=rng.rolling(20,min_periods=10).median(); f=-(intr)*(rng/scale).clip(0,4); fr=x.close.shift(-h)/x.close-1; rr.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}).dropna().reset_index())
    q=pd.concat(rr); a=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
    a=pd.Series(a).dropna(); print('horizon',h,'dates',len(a),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1)*np.sqrt(252))
pv=R.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',pv.diff().abs().mean(axis=1).mean(),'coverage',len(I)/len(pd.date_range(R.date.min(),R.date.max())))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=I.loc[a:b].ic; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252))
