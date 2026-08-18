import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Negative rolling return skewness: assets with recent positive skew are faded, negative skew favored.
px={s:get_stock_daily_data(s,days=2200) for s in U}
rets={}
for s,d in px.items():
    if d is not None and len(d)>100:
        x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date')
        x['r']=x['close'].pct_change(); rets[s]=x.set_index('date')['r']
r=pd.DataFrame(rets).sort_index()
# signal available at t; forward non-overlapping 1d and horizons
sig=-r.rolling(20,min_periods=15).skew()
rows=[]
for h in [1,5,10,20]:
    vals=[]
    for dt in r.index:
        f=sig.loc[dt]; fr=r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt] if False else None
        # forward cumulative return t+1 ... t+h
        ix=r.index.get_loc(dt)
        if ix+h>=len(r): continue
        future=r.iloc[ix+1:ix+h+1].sum(axis=0, min_count=h)
        z=pd.concat([f,future],axis=1).dropna()
        if len(z)>=8:
            ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if np.isfinite(ic): vals.append((dt,ic,len(z)))
    a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    mean=a.ic.mean(); sd=a.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    hit=(a.ic>0).mean()
    print(f'h={h} dates={len(a)} mean_n={a.n.mean():.2f} IC={mean:.6f} ICIR={icir:.6f} hit={hit:.4f}')
# turnover and coverage at daily ranks
valid=sig.notna(); coverage=valid.sum().sum()/valid.size
rank=sig.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
print(f'coverage={coverage:.6f} turnover={turnover:.6f} dates={len(r)} assets={len(r.columns)} cutoff={r.index.max().date()}')
# regime slices for 1d
vals=[]
for dt in r.index:
 ix=r.index.get_loc(dt)
 if ix+1>=len(r): continue
 z=pd.concat([sig.loc[dt],r.iloc[ix+1]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals)).sort_index()
for name,x in [('early',a[a.index<'2023-01-01']),('late',a[a.index>='2025-01-01']),('recent250',a.tail(250))]:
 print(name,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252) if len(x)>2 else np.nan)
