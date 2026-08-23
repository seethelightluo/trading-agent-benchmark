import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d)>100:
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d.close
px=pd.DataFrame(frames).sort_index()
# candidate: agreement-weighted trend, using only completed data and lagging one day
r=px.pct_change()
# compute each asset on its own trading calendar before panel alignment
sigs={}
for c in px.columns:
    q=px[c].dropna(); rr=q.pct_change(); r5=q.pct_change(5); r10=q.pct_change(10); r20=q.pct_change(20)
    vol=rr.rolling(20).std()*np.sqrt(20)
    agree=(np.sign(r5)+np.sign(r10)+np.sign(r20))/3
    sigs[c]=((0.25*r5+0.35*r10+0.40*r20)/(vol+1e-8)*(0.5+0.5*agree)).shift(1)
sig=pd.DataFrame(sigs).reindex(px.index)
# forward close-to-close returns from signal date
out=[]
for h in [1,5,10,20]:
    fwd=px.shift(-h)/px-1
    vals=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
    q=pd.DataFrame(vals,columns=['date','ic','n']).dropna()
    print(f'h={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
# coverage and rank turnover on common dates
print('assets',len(px.columns),'date_range',px.index.min().date(),px.index.max().date())
print('coverage',sig.notna().sum().sum()/(len(sig)*len(U)))
ranks=sig.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
# regime daily 5d
fwd=px.shift(-5)/px-1
vals=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(vals,columns=['date','ic']).dropna(); q.date=pd.to_datetime(q.date)
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31')]:
 t=q[(q.date>=a)&(q.date<=b)]; print(name,len(t),t.ic.mean() if len(t) else np.nan, t.ic.mean()/t.ic.std(ddof=1) if len(t)>1 else np.nan)
