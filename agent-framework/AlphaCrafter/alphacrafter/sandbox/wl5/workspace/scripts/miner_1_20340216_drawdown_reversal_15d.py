import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=get_stock_daily_data(s, days=5200)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date')['close'].astype(float)
        raw[s]=d
px=pd.DataFrame(raw).sort_index().ffill()
# Candidate: 15-day residual reversal, amplified by bounded 60-day drawdown.
r=px.pct_change(15); dd=(px/px.rolling(60,min_periods=60).max()-1).clip(-.5,0)
f=-(r.sub(r.median(axis=1),axis=0))* (1+0.8*(-dd))
# forward return beginning after signal date, 10 trading days
fr=px.shift(-10)/px-1
rows=[]; sig=[]
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        rows.append(a[ok].corr(b[ok],method='spearman'))
        sig.append((dt,a.rank(pct=True)))
ic=np.array(rows,float); ic=ic[np.isfinite(ic)]
print('dates',len(ic),'instruments',len(U),'meanN',len(U),'coverage',float(f.notna().sum().sum()/f.size))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1)*np.sqrt(252/10),'hit',np.mean(ic>0))
# rank turnover, mean adjacent rank displacement
rr=pd.DataFrame({d:x for d,x in sig}).T.sort_index(); turn=(rr.diff().abs().mean(axis=1)/2).mean()
print('turnover',turn)
for h in [5,10,20]:
    x=px.shift(-h)/px-1; vals=[]
    for dt in f.index:
        a=f.loc[dt]; b=x.loc[dt]; ok=a.notna()&b.notna()
        if ok.sum()>=8: vals.append(a[ok].corr(b[ok],method='spearman'))
    print('decay',h,np.nanmean(vals),len(vals))
for name,(a,b) in {'2020-24':('2020-07-01','2024-12-31'),'2025-27':('2025-01-01','2027-12-31'),'2028-29':('2028-01-01','2029-12-31'),'2030-34':('2030-01-01','2034-02-01')}.items():
    z=[]
    for dt in f.loc[a:b].index:
        q=f.loc[dt]; y=fr.loc[dt]; ok=q.notna()&y.notna()
        if ok.sum()>=8:z.append(q[ok].corr(y[ok],method='spearman'))
    print('regime',name,np.nanmean(z),len(z))
# save causal signal artifact
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20340216_drawdown_reversal_15d_signal.csv')
