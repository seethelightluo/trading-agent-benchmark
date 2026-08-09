"""Single-idea validation: continuously dispersion-scaled abnormal-range reversal.
Unlike a binary stress gate, retain the short-horizon exhaustion signal on every
completed date, but scale it by the lagged percentile of cross-asset return
dispersion. This tests whether stress strength is monotonic rather than only
present in the upper dispersion tail. No forward data enter the signal.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-29'); HORIZONS=[1,5,10,20]
cls, sigs = {}, {}
for a in ASSETS:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
    for c in ('high','low','close'): d[c]=pd.to_numeric(d[c],errors='coerce')
    c=d.close.replace(0,np.nan); r=c.pct_change(fill_method=None)
    relative_range=(d.high-d.low).abs().div(c).replace([np.inf,-np.inf],np.nan)
    normal_range=relative_range.rolling(20,min_periods=8).median().shift(1)
    # Smooth completed-bar abnormal range reversal before applying t-1 regime scale.
    sigs[a]=(-r*relative_range.div(normal_range)).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=2).mean()
    cls[a]=c
close=pd.DataFrame(cls); base=pd.DataFrame(sigs).reindex(close.index)
r=close.pct_change(fill_method=None)
dispersion=r.sub(r.median(axis=1),axis=0).abs().median(axis=1)
# Percentile is shifted: decision at t sees dispersion only through t-1.
dispersion_percentile=dispersion.rolling(60,min_periods=30).rank(pct=True).shift(1)
signal=base.mul(dispersion_percentile,axis=0)
print('FACTOR continuous_dispersion_scaled_smoothed_abnormal_range_reversal_3_20_60obs')
print('cutoff',END.date(),'assets',len(ASSETS),'signal_cells',int(signal.notna().sum().sum()),'/',signal.size,'coverage',round(signal.notna().sum().sum()/signal.size,6))
print('dispersion_scale_valid_dates',int(dispersion_percentile.notna().sum()),'mean_scale',round(dispersion_percentile.mean(),6))
all_details={}
for h in HORIZONS:
    fwd=close.pct_change(h,fill_method=None).shift(-h); vals=[]; dates=[]; counts=[]
    for dt in signal.index:
        q=pd.concat([signal.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
            z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(z): vals.append(z);dates.append(dt);counts.append(len(q))
    x=np.array(vals); di=pd.DatetimeIndex(dates); n=np.array(counts)
    ir=x.mean()/x.std(ddof=1) if len(x)>1 else np.nan
    print('H',h,'IC',round(x.mean(),6),'ICIR',round(ir,6),'hit',round((x>0).mean(),6),'dates',len(x),'mean_n',round(n.mean(),3),'min_n',n.min(),'GATE',abs(x.mean())>=.007 and abs(ir)>=.084)
    for label,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_30','2026-01-01',str(END.date()))]:
        z=x[(di>=lo)&(di<=hi)]
        if len(z)>1: print(' REGIME',h,label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rank=signal.rank(axis=1,pct=True); turnover=[]
for t in range(1,len(rank)):
    q=pd.concat([rank.iloc[t-1],rank.iloc[t]],axis=1).dropna()
    if len(q)>=8: turnover.append(np.abs(q.iloc[:,0]-q.iloc[:,1]).mean())
print('rank_turnover',round(float(np.mean(turnover)),6),'adjacent_dates',len(turnover))
print('DECAY: horizons 1,5,10,20 above. Admission remains conditional on complete library signal-correlation evidence.')
signal.to_pickle('scripts/miner_1_20300530_continuous_dispersion_range_reversal_candidate_signal.pkl')
