import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
p=pd.DataFrame(frames).sort_index().ffill()
r=p.pct_change()
m=r.mean(axis=1)
# completed date signal and next calendar available cross-section return
rows=[]
for w in [20,40,60,120]:
    for t in range(w, len(r)-1):
        dt=r.index[t]; nxt=r.index[t+1]
        # only information through t
        win=pd.concat([r.iloc[t-w+1:t+1],m.iloc[t-w+1:t+1].rename('m')],axis=1).dropna()
        neg=win[win.m<0]
        if len(neg)<8: continue
        var=neg.m.var()
        if var<=0: continue
        sig=-(neg.drop(columns='m').covwith if False else pd.Series(index=U,dtype=float))
        # covariance against market on negative-market days, using pairwise available
        vals={}
        for s in U:
            z=neg[[s,'m']].dropna()
            vals[s]=-z[s].cov(z.m)/z.m.var() if len(z)>=8 and z.m.var()>0 else np.nan
        f=pd.Series(vals)
        fr=r.loc[nxt]
        q=pd.concat([f,fr],axis=1).dropna()
        if len(q)>=8:
            ic=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
            rows.append((dt,w,ic,len(q),f))
for w in [20,40,60,120]:
    a=[x for x in rows if x[1]==w]
    ics=pd.Series([x[2] for x in a])
    # rank turnover among consecutive signal snapshots, proxy
    turns=[]
    prev=None
    for x in a:
        ranks=x[4].rank(pct=True)
        if prev is not None: turns.append((ranks-prev).abs().mean())
        prev=ranks
    print('W',w,'dates',len(a),'avg_n',round(np.mean([x[3] for x in a]),2),'IC',round(ics.mean(),6),'ICIR',round(ics.mean()/ics.std(ddof=1),6),'hit',round((ics>0).mean(),4),'turn',round(np.nanmean(turns),4))
    for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('recent','2026-07-16','2027-01-13')]:
        z=[x[2] for x in a if str(x[0])[:10]>=lo and str(x[0])[:10]<=hi]
        if len(z): print(' ',label,len(z),round(np.mean(z),6),round(np.mean(z)/np.std(z,ddof=1),6) if len(z)>1 else np.nan)
print('data',len(p),p.index.min(),p.index.max())
