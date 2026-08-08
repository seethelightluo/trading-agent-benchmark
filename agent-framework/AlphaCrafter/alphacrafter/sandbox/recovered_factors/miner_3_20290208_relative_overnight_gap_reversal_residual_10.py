"""Miner 3: residualized relative overnight-gap reversal, one candidate idea.
Uses only rows through the stated visible cutoff.  A high score identifies assets
whose recent overnight moves have been unusually negative after removing 20d trend
and volatility; cross-asset reversal is tested at several forward horizons.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2029-02-07'); W=10
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in watch:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 p[s]=d.loc[:CUT]
close=pd.concat({s:p[s]['close'] for s in watch},axis=1).sort_index()
opn=pd.concat({s:p[s]['open'] for s in watch},axis=1).reindex(close.index)
r=close.pct_change(); gap=opn/close.shift()-1
# Negative mean gap -> high score.  Standardize per asset window to avoid price-scale effects.
raw=-gap.rolling(W).mean()/gap.rolling(20).std()
trend=close/close.shift(20)-1; vol=r.rolling(20).std()
f=raw.copy()*np.nan
for dt in close.index:
 z=pd.concat([raw.loc[dt],trend.loc[dt],vol.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:].values]
  f.loc[dt,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
def stats(h,rule=lambda d:True):
 fw=close.shift(-h)/close-1; a=[]; ns=[]
 for dt in f.index:
  if rule(dt):
   q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(q)>=8: a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
 a=np.asarray(a); return {'dates':len(a),'ic':round(float(a.mean()),6),'icir':round(float(a.mean()/a.std(ddof=1)),6),'hit':round(float((a>0).mean()),4),'mean_n':round(float(np.mean(ns)),2)}
print('FACTOR residualized 10d relative overnight-gap reversal; cutoff',CUT.date())
print('valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
for h in (1,5,10,20): print('h',h,stats(h))
print('10d_2026_2027',stats(10,lambda d:pd.Timestamp('2026-01-01')<=d<=pd.Timestamp('2027-12-31')))
print('10d_2028_plus',stats(10,lambda d:d>=pd.Timestamp('2028-01-01')))
print('10d_2029',stats(10,lambda d:d>=pd.Timestamp('2029-01-01')))
# Persist candidate signal solely for a separate full-library-correlation audit if predictive gates pass.
f.to_pickle('scripts/miner_3_20290208_gap_reversal_signal.pkl')
