"""Miner 3 research: conditional intraday reversal asymmetry; visible cutoff only."""
import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

CUT='2029-01-24'; W=30; MIN=10
files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
panel={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
 panel[s]=d
# aligned values: use daily close return as market state and open-to-close intraday return as response
close=pd.concat({s:panel[s]['close'] for s in watch},axis=1).sort_index()
opn=pd.concat({s:panel[s]['open'] for s in watch},axis=1).reindex(close.index)
r=close.pct_change(); intra=close/opn-1
mkt=r.mean(axis=1,skipna=True)
# Higher: relative tendency to recover intraday on broad down days rather than give back gains on up days.
f=pd.DataFrame(index=close.index,columns=watch,dtype=float)
for t in range(W,len(close)):
 idx=close.index[t-W:t]; dn=mkt.loc[idx]<0; up=mkt.loc[idx]>0
 if dn.sum()<MIN or up.sum()<MIN: continue
 f.loc[close.index[t]]=intra.loc[idx].where(dn,axis=0).mean()-intra.loc[idx].where(up,axis=0).mean()
# remove 20d trend and 20d realized vol cross-sectionally to target intraday conditional behavior
trend=close/close.shift(20)-1; vol=r.rolling(20).std()
res=f.copy()
for dt in f.index:
 z=pd.concat([f.loc[dt],trend.loc[dt],vol.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1:].values]; b=np.linalg.lstsq(X,z.iloc[:,0].values,rcond=None)[0]
  res.loc[dt,z.index]=z.iloc[:,0]-X@b

def stats(h, subset=None):
 vals=[]; ns=[]
 fw=close.shift(-h)/close-1
 for dt in res.index:
  if subset is not None and not subset(dt): continue
  x=pd.concat([res.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x))
 a=np.array(vals); return dict(dates=len(a),ic=float(a.mean()),icir=float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else np.nan,hit=float((a>0).mean()),mean_n=float(np.mean(ns)))
print('FACTOR: residualized conditional intraday reversal asymmetry, W=30; cutoff',CUT)
print('valid cells',int(res.notna().sum().sum()),'coverage',round(float(res.notna().mean().mean()),4),'mean rank turnover',round(float(res.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
for h in [1,5,10,20]: print('horizon',h,stats(h))
for name,rule in [('2026-2027',lambda d: pd.Timestamp('2026-01-01')<=d<=pd.Timestamp('2027-12-31')),('2028+',lambda d:d>=pd.Timestamp('2028-01-01'))]: print('10d regime',name,stats(10,rule))
# Candidate is only admitted if gate passes; print sufficient correlation evidence vs closest related currently active reconstructed downside-beta signal.
beta=pd.DataFrame(index=close.index,columns=watch,dtype=float)
for t in range(W,len(close)):
 idx=close.index[t-W:t]; dn=mkt.loc[idx]<0; up=mkt.loc[idx]>0
 if dn.sum()>=MIN and up.sum()>=MIN:
  for s in watch:
   a=r.loc[idx,s];
   def bb(mask):
    q=pd.concat([a[mask],mkt.loc[idx][mask]],axis=1).dropna()
    return q.iloc[:,0].cov(q.iloc[:,1])/q.iloc[:,1].var() if len(q)>=MIN and q.iloc[:,1].var()>0 else np.nan
   beta.loc[close.index[t],s]=-(bb(dn)-bb(up))
x=pd.concat([res.stack().rename('a'),beta.stack().rename('b')],axis=1).dropna()
print('related-factor corr downside_beta',float(spearmanr(x.a,x.b).statistic),'common_cells',len(x))
# save machine-readable temporary research output (not a factor library record)
os.makedirs('scripts',exist_ok=True)
