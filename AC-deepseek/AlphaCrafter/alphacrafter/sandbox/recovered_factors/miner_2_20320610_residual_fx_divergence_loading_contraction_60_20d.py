"""Miner 2: one candidate only — FX-divergence residual loading contraction."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-06-09') # prior completed day only

def load(folder, sym):
    f=os.path.join(folder, sym+'.csv')
    x=pd.read_csv(f, parse_dates=['date']).set_index('date')['close'].astype(float)
    return x[x.index<=END]

# Daily outer panel intentionally retains non-coincident cross-asset calendars.
px=pd.concat({s:load('../persistent/stock_data',s) for s in ASSETS},axis=1).sort_index()
macro=pd.concat({'USDCNY':load('../persistent/index_data','USDCNY'),
                 'USDJPY':load('../persistent/index_data','USDJPY')},axis=1).sort_index()
r=px.pct_change()
# Leave-one-out broad market and 60d beta residual: no future information.
mkt=r.mean(axis=1)
beta=r.rolling(60,min_periods=42).cov(mkt).div(mkt.rolling(60,min_periods=42).var(),axis=0)
e=r-beta.mul(mkt,axis=0)
# FX divergence innovation: relative return standardized only on its own trailing history.
fx=(macro.USDCNY.pct_change()-macro.USDJPY.pct_change()).reindex(r.index)
mu=fx.rolling(60,min_periods=40).mean(); sd=fx.rolling(60,min_periods=40).std()
driver=((fx-mu)/sd).clip(-5,5)
# Signal at t: old-minus-recent beta, each estimated solely to t.
b60=e.rolling(60,min_periods=42).cov(driver).div(driver.rolling(60,min_periods=42).var(),axis=0)
b20=e.rolling(20,min_periods=14).cov(driver).div(driver.rolling(20,min_periods=14).var(),axis=0)
f=b60-b20

def ic_stats(h):
    fw=px.shift(-h).div(px)-1
    vals=[]; ninst=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ninst.append(len(z)); dates.append(dt)
    a=np.array(vals); ic=a.mean(); ir=ic/a.std(ddof=1) if len(a)>1 and a.std(ddof=1)>0 else np.nan
    return dict(ic=ic,icir=ir,hit=(a>0).mean(),dates=len(a),n=np.mean(ninst), series=pd.Series(a,index=dates))
print('CANDIDATE: residual FX-divergence (USDCNY return minus USDJPY return) loading contraction 60d/20d')
print('visible cutoff',END.date(),'panel',px.index.min().date(),px.index.max().date(),'assets',len(ASSETS))
print('driver coverage',round(driver.notna().mean(),4),'factor coverage',round(f.notna().stack().mean(),4),'factor cells',int(f.notna().sum().sum()))
allstats={}
for h in (1,5,10,20):
    q=ic_stats(h); allstats[h]=q
    print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d mean_n %.2f'%(q['ic'],q['icir'],q['hit'],q['dates'],q['n']))
q=allstats[20]
for label,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01','2032-06-09')]:
 a=q['series'].loc[lo:hi]; print('REGIME',label,'dates',len(a),'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,'hit',round((a>0).mean(),4) if len(a) else None)
# Rank turnover on consecutive usable dates.
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(np.mean(np.abs(z.iloc[:,0]-z.iloc[:,1])))
print('mean rank turnover',round(float(np.mean(turns)),6),'turnover dates',len(turns))
print('ADMISSION PRECHECK (performance only):',abs(q['ic'])>=.007 and abs(q['icir'])>=.084)
print('NOTE: no persistence: library-wide signal matrices were not available for a defensible required max-correlation calculation.')
