"""Miner_3 research: 20-day close-location pressure (one factor idea).
Uses only records visible through the supplied simulation cutoff.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT='../persistent/stock_data'; cutoff=pd.Timestamp('2033-01-19')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Close-location pressure: average signed position of close within daily range,
# volume-independent and distinct from return momentum/volatility.
panel={}
for a in assets:
    x=pd.read_csv(f'{ROOT}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
    x=x.loc[x.index<=cutoff]
    rng=(x.high-x.low).replace(0,np.nan)
    loc=(2*(x.close-x.low)/rng-1).clip(-1,1)
    # reliability-weighted mean prevents a one-day extreme from dominating
    panel[a]=(loc.rolling(20,min_periods=16).mean()*np.sqrt(20)).rename(a)
factor=pd.DataFrame(panel)
prices=pd.DataFrame({a:pd.read_csv(f'{ROOT}/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cutoff,'close'] for a in assets})
# strict common calendar preserves genuine cross-asset availability
factor=factor.reindex(prices.index)
print('CANDIDATE close_location_pressure_20d; cutoff',cutoff.date())
print('factor dates',factor.dropna(how='all').shape[0],'cells',int(factor.notna().sum().sum()),'coverage',round(factor.notna().mean().mean(),6))
all_ic={}
for h in [1,3,5,7,10,20]:
    fwd=prices.shift(-h)/prices-1
    vals=[]; ns=[]
    for dt in factor.index:
        z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
    v=np.array(vals); ic=float(np.mean(v)); sd=float(np.std(v,ddof=1)); ir=ic/sd if sd else np.nan
    all_ic[h]=(ic,ir,len(v),float(np.mean(v>0)),float(np.mean(ns)))
    print(f'h={h:2d} IC={ic:+.6f} ICIR={ir:+.6f} dates={len(v)} hit={np.mean(v>0):.4f} meanN={np.mean(ns):.2f}')
# turnover rank correlation
turn=[]
for i in range(1,len(factor)):
 z=pd.concat([factor.iloc[i-1],factor.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('rank_turnover',round(float(np.mean(turn)),6),'adjacent_dates',len(turn))
# chronological 10d regimes
fwd=prices.shift(-10)/prices-1
for name,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01','2033-01-19')]:
 v=[]
 for dt in factor.loc[lo:hi].index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 v=np.array(v); print(name,'dates',len(v),'IC',('NA' if not len(v) else f'{v.mean():+.6f}'),'ICIR',('NA' if len(v)<2 or v.std(ddof=1)==0 else f'{v.mean()/v.std(ddof=1):+.6f}'),'hit',('NA' if not len(v) else f'{np.mean(v>0):.4f}'))
# Save signal for a subsequent only-if-gated library-correlation screen; never persistence.
factor.to_pickle('scripts/miner_3_20330120_close_location_pressure_20d_signal.pkl')
print('RESULT10',all_ic[10])
