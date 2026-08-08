import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# One idea: dispersion-conditioned 20-session range-position reversal.
# At high cross-asset dispersion, rank assets by negative close location in their own 20d high-low range.
END='2029-03-21'; assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; hi={}; lo={}
for a in assets:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 d=d.loc[:END]; cl[a]=d.close; hi[a]=d.high; lo[a]=d.low
close=pd.DataFrame(cl).sort_index(); high=pd.DataFrame(hi).reindex(close.index); low=pd.DataFrame(lo).reindex(close.index)
r=close.pct_change(); disp=r.rolling(5,min_periods=5).std(axis=1).rolling(60,min_periods=40).apply(lambda x: x.iloc[-1]/x.median() if x.median()>0 else np.nan,raw=False)
pos=(close-low.rolling(20,min_periods=20).min())/(high.rolling(20,min_periods=20).max()-low.rolling(20,min_periods=20).min())
# high dispersion condition; reversal direction (low range location ranks high)
sig=(0.5-pos).where(disp>=1.0,0.0)

def stats(s,h):
 vals=[]; ns=[]
 fwd=close.shift(-h)/close-1
 for t in s.index:
  x=s.loc[t]; y=fwd.loc[t]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:
   vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 a=np.array(vals); return dict(ic=float(a.mean()),icir=float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else np.nan,hit=float((a>0).mean()),dates=len(a),n=float(np.mean(ns)))
print('idea dispersion-conditioned range-position reversal; endpoint',END)
print('signal coverage',float(sig.notna().mean().mean()),'active share',float((sig.abs().sum(1)>0).mean()),'dates',len(sig))
for h in [1,5,10,20]: print('horizon',h,stats(sig,h))
# turnover/rank stability active pair observations
cors=[]
for i in range(1,len(sig)):
 a,b=sig.iloc[i-1],sig.iloc[i]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1: cors.append(spearmanr(a[ok],b[ok]).statistic)
print('rank stability',np.nanmean(cors),'turnover',1-np.nanmean(cors),'pairs',len(cors))
# regimes 5d
for name,mask in [('2026_2028',sig.index<='2028-12-31'),('2029_ytd',sig.index>='2029-01-01')]: print(name,stats(sig.loc[mask],5))
# Store candidate first, then assess dated signal artifact panels for ALL effective files
out='scripts/miner_2_20290322_dispersion_conditioned_range_position_reversal_20obs_signal.pkl'; sig.to_pickle(out)
effective=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f));
  if j.get('validation',{}).get('status')=='EFFECTIVE': effective.append(j['factor_id'])
 except: pass
print('effective count',len(effective),effective)
maxrho=0.; evidence=[]; missing=[]
for fid in effective:
 matches=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not matches: missing.append(fid); continue
 try:
  x=pd.read_pickle(matches[-1]);
  if isinstance(x,pd.Series): x=x.unstack() if isinstance(x.index,pd.MultiIndex) else None
  if not isinstance(x,pd.DataFrame): missing.append(fid);continue
  # normalize index and columns overlap
  x.index=pd.to_datetime(x.index); commoni=sig.index.intersection(x.index); commonc=sig.columns.intersection(x.columns)
  rs=[]
  for t in commoni:
   a=sig.loc[t,commonc]; b=x.loc[t,commonc]; ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1: rs.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if rs: evidence.append((fid,max(rs),len(rs))); maxrho=max(maxrho,max(rs))
  else: missing.append(fid)
 except Exception as e: missing.append(fid)
print('correlation evidence',evidence)
print('max_abs_library_correlation',maxrho,'missing',missing)
