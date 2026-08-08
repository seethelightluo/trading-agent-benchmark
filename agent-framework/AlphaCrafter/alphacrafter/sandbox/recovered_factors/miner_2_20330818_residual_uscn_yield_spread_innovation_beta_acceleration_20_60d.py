"""miner_2: validate one idea -- residual US/CN yield-spread innovation beta acceleration.
All inputs are truncated at the last completed day before 2033-08-18.
"""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2033-08-17')
# Outer-date panel preserves each instrument's actual completed observations.
closes={}
for a in ASSETS:
    x=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date')['close']
    closes[a]=x.loc[x.index<=CUT]
px=pd.DataFrame(closes).sort_index()
r=px.pct_change(fill_method=None)
# Observation-only US/CN yield spread; contemporaneous driver is available at decision close.
def macro(sym):
    x=pd.read_csv(f'../persistent/index_data/{sym}.csv',parse_dates=['date']).set_index('date')
    col='close' if 'close' in x else x.select_dtypes('number').columns[0]
    return x[col].loc[lambda z:z.index<=CUT]
us=macro('US10Y'); cn=macro('CN10Y')
spread=(us-cn).reindex(px.index).ffill()
innov=spread.diff()
driver=(innov-innov.rolling(60,min_periods=40).mean())/innov.rolling(60,min_periods=40).std(ddof=0)
driver=driver.clip(-5,5)
# Residualize returns against equal-weight market return before estimating yield-spread exposure.
mkt=r.mean(axis=1,skipna=True)
res=pd.DataFrame(index=r.index,columns=ASSETS,dtype=float)
for a in ASSETS:
    # rolling beta to market and residual, with a sufficiently long local estimate
    b=r[a].rolling(60,min_periods=40).cov(mkt)/mkt.rolling(60,min_periods=40).var()
    res[a]=r[a]-b*mkt
# beta20-beta60: recent strengthening in idiosyncratic sensitivity to bilateral yield divergence
b20=res.rolling(20,min_periods=15).cov(driver).div(driver.rolling(20,min_periods=15).var(),axis=0)
b60=res.rolling(60,min_periods=40).cov(driver).div(driver.rolling(60,min_periods=40).var(),axis=0)
f=b20-b60

def ic_series(h):
    fw=px.shift(-h).div(px)-1
    vals=[]; sizes=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)); sizes.append(len(z))
    return pd.Series(dict(vals),dtype=float),sizes
print('IDEA residual US-CN yield-spread innovation beta acceleration (20d minus 60d)')
print('cutoff',CUT.date(),'universe',len(ASSETS),'panel dates',len(px),'signal cells',int(f.notna().sum().sum()),'coverage',round(f.notna().mean().mean(),6),'driver coverage',round(driver.notna().mean(),6))
allics={}
for h in (1,5,10,20):
    s,n=ic_series(h); allics[h]=s
    print(f'h={h:2d} IC={s.mean():+.6f} ICIR={s.mean()/s.std(ddof=1):+.6f} hit={(s>0).mean():.4f} dates={len(s)} mean_n={np.mean(n):.2f}')
# Predefined multi-regime readout at strongest conventional 20d horizon.
s=allics[20]
for name,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_plus','2027-01-01','2033-08-17')]:
 q=s.loc[lo:hi]
 print('regime',name,'dates',len(q),'IC',('NA' if len(q)<2 else f'{q.mean():+.6f}'),'ICIR',('NA' if len(q)<2 else f'{q.mean()/q.std(ddof=1):+.6f}'),'hit',('NA' if not len(q) else f'{(q>0).mean():.4f}'))
# Rank turnover.
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('rank_turnover',round(float(np.mean(turn)),6),'adjacent_dates',len(turn))
# No stored signal matrices are part of the library JSON schema. Hence a max-library-rho cannot be evidenced safely.
print('LIBRARY_CORRELATION_EVIDENCE=UNAVAILABLE: factor JSON definitions contain no historical signal matrices; admission must fail absent reproducible signals for every admitted factor.')
print('validation_date','2033-08-18')
PY
python scripts/miner_2_20330818_residual_uscn_yield_spread_innovation_beta_acceleration_20_60d.py