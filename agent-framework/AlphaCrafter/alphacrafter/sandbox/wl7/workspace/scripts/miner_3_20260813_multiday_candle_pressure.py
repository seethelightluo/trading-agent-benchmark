import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            d=fn(s,2400)
            if d is not None and len(d)>100: return d
        except Exception: pass
    return None
Ds={s:get(s) for s in U}; Ds={s:d for s,d in Ds.items() if d is not None}
# Multi-day candle pressure: volume-free aggregate CLV weighted by true range, intentionally distinct from 1d CLV
F={}; R={}
for s,d in Ds.items():
    d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize(); d=d.drop_duplicates('date').set_index('date').sort_index()
    h=d.high.astype(float); l=d.low.astype(float); c=d.close.astype(float)
    rng=(h-l).replace(0,np.nan)
    clv=((2*c-h-l)/rng).clip(-1,1)
    # 5-day pressure, recent bars weighted by range; lag one day in evaluation naturally via shift
    f=(clv*(h-l)).rolling(5,min_periods=4).sum()/(h-l).rolling(5,min_periods=4).sum()
    F[s]=f; R[s]=c.pct_change()
all_dates=sorted(set().union(*[set(x.index) for x in F.values()]))
ics=[]; h5=[]; h10=[]; ranks=[]; nobs=[]
for dt in all_dates:
    vals={s:F[s].loc[dt] for s in F if dt in F[s].index and pd.notna(F[s].loc[dt])}
    # next valid observation return (asynchronous calendars), only information through dt
    y1={}; y5={}; y10={}
    for s in vals:
        ix=R[s].index; p=ix.get_loc(dt)
        for k,out in [(1,y1),(5,y5),(10,y10)]:
            if p+k<len(ix): out[s]=R[s].iloc[p+1:p+k+1].sum()
    def corr(y):
        z=[(vals[s],y[s]) for s in vals if s in y and np.isfinite(y[s])]
        return pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z])) if len(z)>=8 else np.nan
    q=corr(y1)
    if pd.notna(q):
        ics.append(q); h5.append(corr(y5)); h10.append(corr(y10)); nobs.append(len(vals))
        ranks.append(pd.Series(vals).rank(pct=True))
# rank turnover
turn=[]
for a,b in zip(ranks[:-1],ranks[1:]):
    common=a.index.intersection(b.index)
    if len(common)>=8: turn.append(np.mean(abs(a[common]-b[common])))
def stat(x):
 x=np.array(x,dtype=float); x=x[np.isfinite(x)]; return (len(x),np.mean(x),np.mean(x)/np.std(x,ddof=1))
print('dates',len(ics),'avg_names',np.mean(nobs),'coverage_dates',len(ics)/len(all_dates))
print('daily n mean ic icir',stat(ics),'hit',np.mean(np.array(ics)>0),'turn',np.mean(turn))
print('5d',stat(h5),'10d',stat(h10))
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 z=[v for dt,v in zip([d for d in all_dates if d in all_dates],ics) if a<=dt.year<=b]
 print('regime',a,b,stat(z))
print('signals',len(F), 'dates',min(all_dates),max(all_dates))
