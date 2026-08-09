# Single-idea validation: drawdown recovery efficiency
import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# close panel; outer alignment preserves native calendars and avoids filling returns across gaps
panels={}
for a in ASSETS:
    f='../persistent/stock_data/'+a+'.csv'
    d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
    panels[a]=d
close=pd.DataFrame(panels).sort_index()
# Each asset signal uses its native last 60 completed price observations. Signal: recent recovery
# from its 60-observation peak drawdown: 20-observation return / absolute current drawdown.
sig=pd.DataFrame(index=close.index,columns=ASSETS,dtype=float)
for a in ASSETS:
    x=close[a]
    peak=x.rolling(60,min_periods=60).max()
    dd=x/peak-1
    ret20=x/x.shift(20)-1
    sig[a]=ret20/(-dd+0.01)  # fixed 1% stabilizer avoids unbounded values at a high
    sig.loc[(dd>=-1e-10) | ~np.isfinite(sig[a]),a]=np.nan
# Cross-section rank IC on dates; returns only available strictly after signal date.
print('candidate=drawdown_recovery_efficiency_20_60obs cutoff=',close.dropna(how='all').index.max().date())
print('signal cells',int(sig.notna().sum().sum()),'of',sig.size, 'coverage',round(sig.notna().sum().sum()/sig.size,4))
results={}
for h in [1,5,10,20]:
    vals=[]; counts=[]; dates=[]
    # forward return by each asset's native h observations, then attach date; no future signal use
    fwd=close.apply(lambda x:x.shift(-h)/x-1)
    for dt in sig.index:
        q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(q)>=8:
            ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(ic): vals.append(ic); counts.append(len(q)); dates.append(dt)
    v=np.array(vals); ir=v.mean()/v.std(ddof=1) if len(v)>1 and v.std(ddof=1)>0 else np.nan
    print('h',h,'IC',round(v.mean(),6),'ICIR',round(ir,6),'hit',round((v>0).mean(),4),'dates',len(v),'mean_n',round(np.mean(counts),2),'min_n',min(counts) if counts else None)
    # calendar regime slices
    for label,lo,hi in [('2020-2021','2020-01-01','2021-12-31'),('2022-2023','2022-01-01','2023-12-31'),('2024-2025','2024-01-01','2025-12-31'),('2026-2030','2026-01-01','2030-12-31')]:
        z=np.array([x for x,d in zip(vals,dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
        if len(z)>1: print(' ',label,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
    results[h]=(v,dates)
# Mean absolute percentile-rank signal movement, only paired names, native daily calendar
turn=[]
for i in range(1,len(sig)):
    a=sig.iloc[i-1];b=sig.iloc[i];q=pd.concat([a,b],axis=1).dropna()
    if len(q)>=8: turn.append(np.mean(np.abs(q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True))))
print('rank_turnover',round(float(np.mean(turn)),6),'adjacent_dates',len(turn))
sig.to_pickle('scripts/miner_1_drawdown_recovery_efficiency_candidate_signal.pkl')
