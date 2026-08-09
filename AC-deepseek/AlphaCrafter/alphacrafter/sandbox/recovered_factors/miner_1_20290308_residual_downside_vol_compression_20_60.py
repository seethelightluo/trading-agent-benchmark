"""Miner 1: validate one idea -- residual idiosyncratic downside-volatility compression."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CUT = pd.Timestamp('2029-03-07')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# One idea: a falling short/long downside residual-vol ratio should identify assets
# whose idiosyncratic left-tail risk is normalising.  We remove trend and total vol.
close={}
for a in ASSETS:
    p=f'../persistent/stock_data/{a}.csv'
    d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
    close[a]=d[d.index<=CUT]
P=pd.DataFrame(close).sort_index(); R=P.pct_change()
M=R.mean(axis=1,skipna=True)
# 60d rolling beta residuals, requiring broad contemporaneous market data
B=R.rolling(60,min_periods=40).cov(M).unstack().reindex(columns=ASSETS).div(M.rolling(60,min_periods=40).var(),axis=0)
E=R-B.mul(M,axis=0)
D=E.where(E<0,0.0)
# downside RMS is used rather than ordinary vol, so this is an explicit tail-risk variant
short=np.sqrt((D*D).rolling(20,min_periods=15).mean())
long=np.sqrt((D*D).rolling(60,min_periods=40).mean())
raw=-(short/(long+1e-12))
trend=(P/P.shift(20)-1)/R.rolling(20,min_periods=15).std()
vol=R.rolling(20,min_periods=15).std()
# daily cross-sectional OLS residual; at least eight of 15 assets
F=pd.DataFrame(index=P.index,columns=ASSETS,dtype=float)
for t in P.index:
    z=pd.concat([raw.loc[t],trend.loc[t],vol.loc[t]],axis=1).dropna()
    if len(z)>=8:
        y=z.iloc[:,0].values; X=np.column_stack([np.ones(len(z)),z.iloc[:,1:].values])
        F.loc[t,z.index]=y-X@np.linalg.lstsq(X,y,rcond=None)[0]

def stats(h):
    ics=[]; ns=[]
    fw=P.shift(-h)/P-1
    for t in F.index:
        z=pd.concat([F.loc[t],fw.loc[t]],axis=1).dropna()
        if len(z)>=8:
            ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
    x=np.array(ics); return dict(ic=float(x.mean()),icir=float(x.mean()/x.std(ddof=1)) if len(x)>1 else np.nan,
        hit=float((x>0).mean()),dates=len(x),mean_n=float(np.mean(ns)),std=float(x.std(ddof=1)))
print('IDEA residual_idiosyncratic_downside_vol_compression_20_60')
print('cutoff',CUT.date(),'panel_dates',len(P),'assets',len(ASSETS))
for h in (1,5,10,20): print('H',h,stats(h))
# Robustness by non-overlapping calendar regimes at the selected 10d horizon
fw=P.shift(-10)/P-1
for name,lo,hi in [('2020_2022','2020-01-01','2022-12-31'),('2023_2025','2023-01-01','2025-12-31'),('2026_2029','2026-01-01','2029-02-20')]:
  ics=[]
  for t in F.loc[lo:hi].index:
    z=pd.concat([F.loc[t],fw.loc[t]],axis=1).dropna()
    if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=np.array(ics); print('REGIME',name,'dates',len(x),'ic',round(float(x.mean()),6) if len(x) else None,'icir',round(float(x.mean()/x.std(ddof=1)),6) if len(x)>1 else None,'hit',round(float((x>0).mean()),4) if len(x) else None)
# Rank turnover and coverage
turn=[]
for t0,t1 in zip(F.index[:-1],F.index[1:]):
 z=pd.concat([F.loc[t0],F.loc[t1]],axis=1).dropna()
 if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('coverage',round(float(F.notna().sum().sum()/F.size),6),'turnover',round(float(np.mean(turn)),6),'turnover_dates',len(turn))
# Independence evidence against closest admitted conceptual factor: realized-vol compression,
# computed exactly as its persisted expression on the same asset/date cells.
VC=-(R.rolling(20,min_periods=15).std()/R.rolling(60,min_periods=40).std())
z=pd.concat([F.stack().rename('new'),VC.stack().rename('volcomp')],axis=1).dropna()
print('proxy_library_corr_realized_vol_compression',round(float(spearmanr(z.new,z.volcomp).statistic),6),'common_cells',len(z))
# Save results only to stdout; admission needs exact all-library correlation evidence.
