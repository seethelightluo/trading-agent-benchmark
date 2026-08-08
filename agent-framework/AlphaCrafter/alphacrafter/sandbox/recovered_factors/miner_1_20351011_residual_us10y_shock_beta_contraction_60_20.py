import os, json, glob, re
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
# Single candidate: residual sensitivity contraction to US 10-year yield shocks.
# High values mean an asset's idiosyncratic return has become less sensitive to a
# standardized yield move over the recent 20 sessions than over its 60-session baseline.
N=5000
assets=list(get_account_dict()['watch_list'])
def frame(a):
    x=get_stock_daily_data(a,N).copy(); x['date']=pd.to_datetime(x['date'])
    return x.set_index('date').sort_index()['close']
close=pd.DataFrame({a:frame(a) for a in assets}).sort_index()
r=close.pct_change()
market=r.mean(axis=1)
# visibility-safe rolling market residuals
vm=market.rolling(60,min_periods=42).var()
beta_m=r.rolling(60,min_periods=42).cov(market).div(vm,axis=0)
resid=r.sub(beta_m.mul(market,axis=0))
yshock=r['US10Y']
# Standardize only with history through t.  Since US10Y is an intentional tradable series,
# using its completed daily move as an observed cross-asset shock is allowed.
z=(yshock-yshock.rolling(60,min_periods=42).mean()).div(yshock.rolling(60,min_periods=42).std())
v20=z.rolling(20,min_periods=16).var(); v60=z.rolling(60,min_periods=42).var()
b20=resid.rolling(20,min_periods=16).cov(z).div(v20,axis=0)
b60=resid.rolling(60,min_periods=42).cov(z).div(v60,axis=0)
raw=b60-b20
# Remove contemporaneous residual trend and lower-tail variability cross-sectionally.
mom=resid.rolling(20,min_periods=16).mean(); semi=resid.where(resid<0).pow(2).rolling(20,min_periods=16).mean().pow(.5)
sig=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for dt in sig.index:
    q=pd.concat([raw.loc[dt].rename('raw'),mom.loc[dt].rename('mom'),semi.loc[dt].rename('semi')],axis=1).dropna()
    if len(q)>=8:
      X=np.c_[np.ones(len(q)),q[['mom','semi']].to_numpy()]
      sig.loc[dt,q.index]=q.raw-X@np.linalg.lstsq(X,q.raw.to_numpy(),rcond=None)[0]
def stat(h,lo=None,hi=None):
 fwd=close.shift(-h).div(close)-1; out=[]
 ix=sig.loc[lo:hi].index if lo else sig.index
 for d in ix:
  q=pd.concat([sig.loc[d].rename('s'),fwd.loc[d].rename('f')],axis=1).dropna()
  if len(q)>=8: out.append(q.s.corr(q.f,method='spearman'))
 a=np.array(out,float); return (len(a),float(a.mean()) if len(a) else np.nan,float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else np.nan,float((a>0).mean()) if len(a) else np.nan)
print('FACTOR residual_us10y_shock_beta_contraction_60_20')
print('cutoff',close.index.max().date(),'assets',len(assets),'signal_dates',int(sig.notna().any(axis=1).sum()),'valid_cells',int(sig.notna().sum().sum()),'coverage',round(float(sig.notna().mean().mean()),6),'mean_names',round(float(sig.notna().sum(axis=1).mean()),3))
for h in [1,5,10,20]: print('H',h,'n_ic ic icir hit',stat(h))
ranks=sig.rank(axis=1,pct=True); zz=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0)
print('turnover',round(float(ranks.diff().abs().mean(axis=1).mean()),6),'concentration',round(float(zz.abs().stack().mean()),6))
for a,b,label in [('2020-01-01','2024-12-31','2020_24'),('2025-01-01','2029-12-31','2025_29'),('2030-01-01','2034-12-31','2030_34'),('2035-01-01','2100-01-01','2035_ytd')]: print('REGIME',label,'H5',stat(5,a,b))
# Resolve a panel for every EFFECTIVE library record via factor-id suffix, then require data evidence.
def norm(x): return re.sub(r'[^a-z0-9]','',x.lower())
active=[]
for f in glob.glob('factors/*.json'):
 try:
  d=json.load(open(f))
  if d.get('validation',{}).get('status')=='EFFECTIVE': active.append((f,d.get('factor_id','')))
 except Exception: pass
panels=glob.glob('scripts/*signal.pkl')
resolved=[]; missing=[]
for f,fid in active:
 key=fid.split('_',2)[-1] if fid.startswith('miner_') else fid
 nk=norm(key)
 candidates=[p for p in panels if nk in norm(os.path.basename(p))]
 if not candidates:
  # deterministic overlap score allows dated panel names differing only by a descriptive prefix
  toks=[t for t in re.split(r'[_-]',key) if len(t)>=5 and not t.isdigit()]
  scored=[(sum(norm(t) in norm(os.path.basename(p)) for t in toks),p) for p in panels]
  candidates=[max(scored)[1]] if scored and max(scored)[0]>=max(2,len(toks)//2) else []
 if candidates:
  try:
   o=pd.read_pickle(sorted(candidates)[-1]); q=pd.concat([sig.stack().rename('x'),o.stack().rename('y')],axis=1).dropna()
   if len(q)>=100: resolved.append((abs(q.x.corr(q.y,method='spearman')),fid,len(q),os.path.basename(sorted(candidates)[-1])))
   else: missing.append(fid+'(insufficient-common)')
  except Exception: missing.append(fid+'(unreadable)')
 else: missing.append(fid)
print('LIBRARY active',len(active),'resolved',len(resolved),'missing',len(missing))
print('LIBRARY max',max(resolved) if resolved else None)
print('LIBRARY top5',sorted(resolved,reverse=True)[:5])
print('LIBRARY missing_ids',missing)
sig.to_pickle('scripts/miner_1_20351011_residual_us10y_shock_beta_contraction_60_20_signal.pkl')
