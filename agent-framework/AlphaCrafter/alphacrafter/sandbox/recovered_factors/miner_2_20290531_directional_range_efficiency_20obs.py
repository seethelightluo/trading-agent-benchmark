import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# One idea: directionally conditioned range-to-return efficiency.  A high score
# means a 20-session directional move occurred with little intraday range per
# unit return, i.e. efficient trend rather than noisy price discovery.
END='2029-05-30'
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END] for a in ASSETS}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
rng=pd.DataFrame({a:(x['high']-x['low'])/x['close'] for a,x in raw.items()}).sort_index()
ret=close.pct_change()
# Signed 20d displacement normalized by 20d mean intraday range.  The cross-
# sectional signal keeps the directional conditioning, but rewards efficient moves.
sig=(close/close.shift(20)-1)/(rng.rolling(20,min_periods=15).mean()*np.sqrt(20))
def calc(s,h,mask=None):
 fwd=close.shift(-h)/close-1; vals=[]; ns=[]
 dates=s.index if mask is None else s.index[np.asarray(mask,dtype=bool)]
 for t in dates:
  x=s.loc[t];y=fwd.loc[t];ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
   vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
 z=np.asarray(vals,float); sd=z.std(ddof=1) if len(z)>1 else np.nan
 return {'ic':float(z.mean()) if len(z) else None,'icir':float(z.mean()/sd) if sd>0 else None,'hit_ratio':float((z>0).mean()) if len(z) else None,'ic_dates':len(z),'mean_valid_instruments':float(np.mean(ns)) if ns else 0.0}
print('idea=directionally conditioned range-to-return efficiency 20obs; endpoint='+END)
print('PANEL',json.dumps({'dates':len(sig),'coverage':float(sig.notna().mean().mean()),'mean_valid_instruments':float(sig.notna().sum(1).mean()),'latest_valid':str(sig.dropna(how='all').index.max().date())}))
for h in [1,5,10,20]:print('HORIZON',h,json.dumps(calc(sig,h)))
for label,mask in [('2020_2022',sig.index<='2022-12-31'),('2023_2025',(sig.index>='2023-01-01')&(sig.index<='2025-12-31')),('2026_2028',(sig.index>='2026-01-01')&(sig.index<='2028-12-31')),('2029_ytd',sig.index>='2029-01-01')]:print('REGIME',label,json.dumps(calc(sig,1,mask)))
rs=[]
for i in range(1,len(sig)):
 a,b=sig.iloc[i-1],sig.iloc[i];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:rs.append(spearmanr(a[ok],b[ok]).statistic)
print('TURNOVER',json.dumps({'daily_rank_stability':float(np.mean(rs)),'implied_daily_rank_turnover':float(1-np.mean(rs)),'pairs':len(rs)}))
out='scripts/miner_2_20290531_directional_range_efficiency_20obs_signal.pkl';sig.to_pickle(out)
effective=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f))
  if not f.endswith('.bak') and j.get('validation',{}).get('status')=='EFFECTIVE':effective.append(j['factor_id'])
 except Exception:pass
maxrho=0.;ev=[];missing=[]
for fid in effective:
 paths=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not paths:missing.append(fid);continue
 try:
  old=pd.read_pickle(sorted(paths)[-1]);old.index=pd.to_datetime(old.index);vs=[]
  for t in sig.index.intersection(old.index):
   common=sig.columns.intersection(old.columns);a=sig.loc[t,common];b=old.loc[t,common];ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:vs.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if vs:
   mx=float(max(vs));ev.append((fid,mx,len(vs)));maxrho=max(maxrho,mx)
  else:missing.append(fid)
 except Exception:missing.append(fid)
print('LIBRARY_CORRELATION',json.dumps({'effective':len(effective),'audited':len(ev),'max_abs_library_correlation':maxrho,'evidence':ev,'missing':missing}))
