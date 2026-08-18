import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            d=fn(s,2400)
            if d is not None and len(d)>100:return d
        except Exception: pass
    return None
raw={s:get(s) for s in U}; raw={s:d for s,d in raw.items() if d is not None}
R={}
for s,d in raw.items():
    d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize(); d=d.drop_duplicates('date').set_index('date').sort_index()
    R[s]=d.close.astype(float).pct_change()
bench=R['US10Y']; var=bench.rolling(60,min_periods=45).var()
F={}
for s,r in R.items():
    # defensive rate-sensitivity: high score for assets whose returns are low when US10Y rises
    F[s]=-(r.rolling(60,min_periods=45).cov(bench)/var)
all_dates=sorted(set().union(*[set(x.index) for x in F.values()]))
ics={1:[],5:[],10:[]}; nobs=[]; ranks=[]; dated=[]
for dt in all_dates:
    vals={s:F[s].loc[dt] for s in F if dt in F[s].index and pd.notna(F[s].loc[dt])}
    ys={k:{} for k in ics}
    for s in vals:
        ix=R[s].index
        if dt not in ix: continue
        p=ix.get_loc(dt)
        for k in ys:
            if p+k<len(ix): ys[k][s]=R[s].iloc[p+1:p+k+1].sum()
    good={}
    for k in ys:
        z=[(vals[s],ys[k][s]) for s in vals if s in ys[k] and np.isfinite(ys[k][s])]
        if len(z)>=8: ics[k].append(pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z])))
    if len(ics[1])>len(dated):
        nobs.append(len(vals)); dated.append(dt); ranks.append(pd.Series(vals).rank(pct=True))
def stat(a):
 a=np.asarray(a,float); a=a[np.isfinite(a)]; return len(a),float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1))
turn=[]
for a,b in zip(ranks[:-1],ranks[1:]):
 c=a.index.intersection(b.index)
 if len(c)>=8:turn.append(float(np.mean(abs(a[c]-b[c]))))
print('assets',len(R),'all_dates',len(all_dates),'dates',len(ics[1]),'avg_names',float(np.mean(nobs)))
print('coverage',len(ics[1])/len(all_dates),'turnover',float(np.mean(turn)))
for k in ics: print('horizon',k,'n ic icir hit',stat(ics[k]),float(np.mean(np.asarray(ics[k])>0)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=[ics[1][i] for i,d in enumerate(dated) if lo<=d.year<=hi]
 print('regime',lo,hi,stat(z))
print('last_date',max(dated).date())
print('library_corr','unavailable_without_signal_artifacts')
