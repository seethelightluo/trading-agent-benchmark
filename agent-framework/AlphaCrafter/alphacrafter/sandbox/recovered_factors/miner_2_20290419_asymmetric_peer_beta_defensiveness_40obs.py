import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# One idea: asymmetric peer-beta defensiveness. Assets whose beta to the daily
# equal-weighted peer market is lower in peer-down sessions than in peer-up
# sessions are favored. This measures conditional dependence, rather than level
# beta, trend, or a single-asset return path.
END='2029-04-18'
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'] for a in ASSETS}).sort_index()
ret=close.pct_change()
# Leave-one-out market ensures an asset does not mechanically enter its benchmark.
sig=pd.DataFrame(index=close.index,columns=ASSETS,dtype=float)
for k in range(39,len(ret)):
    w=ret.iloc[k-39:k+1]
    for a in ASSETS:
        peers=[p for p in ASSETS if p!=a]
        m=w[peers].mean(axis=1); x=w[a]
        down=(m<0)&m.notna()&x.notna(); up=(m>=0)&m.notna()&x.notna()
        # Require balanced conditional samples. beta is cov/var; score is beta_up-beta_down.
        if down.sum()>=10 and up.sum()>=10 and m[down].var()>0 and m[up].var()>0:
            bd=x[down].cov(m[down])/m[down].var()
            bu=x[up].cov(m[up])/m[up].var()
            sig.loc[sig.index[k],a]=bu-bd

def calc(s,h,mask=None):
    fwd=close.shift(-h)/close-1
    rows=[]; nums=[]
    dates=s.index if mask is None else s.index[mask.reindex(s.index,fill_value=False)]
    for t in dates:
        x=s.loc[t]; y=fwd.loc[t]; ok=x.notna()&y.notna()
        if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1:
            rows.append(spearmanr(x[ok],y[ok]).statistic); nums.append(ok.sum())
    z=np.array(rows,float)
    return {'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)) if len(z)>1 and z.std(ddof=1)>0 else None,'hit_ratio':float((z>0).mean()),'ic_dates':int(len(z)),'mean_valid_instruments':float(np.mean(nums)) if nums else 0.0}
print('idea=asymmetric peer-beta defensiveness; endpoint='+END)
print('signal_panel_dates',len(sig),'coverage',float(sig.notna().mean().mean()),'latest_valid',sig.dropna(how='all').index.max().date())
for h in [1,5,10,20]: print('HORIZON',h,json.dumps(calc(sig,h)))
for label,mask in [('2026_2028',sig.index<='2028-12-31'),('2029_ytd',sig.index>='2029-01-01')]: print('REGIME',label,json.dumps(calc(sig,5,mask)))
# rank turnover
rs=[]
for i in range(1,len(sig)):
 a,b=sig.iloc[i-1],sig.iloc[i]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1: rs.append(spearmanr(a[ok],b[ok]).statistic)
print('TURNOVER',json.dumps({'daily_rank_stability':float(np.mean(rs)),'implied_daily_rank_turnover':float(1-np.mean(rs)),'pairs':len(rs)}))
out='scripts/miner_2_20290419_asymmetric_peer_beta_defensiveness_40obs_signal.pkl';sig.to_pickle(out)
# Contract audit: every EFFECTIVE factor requires signal evidence, and use latest matching artifact.
effective=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f))
  if j.get('validation',{}).get('status')=='EFFECTIVE' and not f.endswith('.bak'): effective.append(j['factor_id'])
 except Exception: pass
maxrho=0.; evidence=[]; missing=[]
for fid in effective:
 paths=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not paths: missing.append(fid); continue
 try:
  old=pd.read_pickle(sorted(paths)[-1]); old.index=pd.to_datetime(old.index)
  vals=[]
  for t in sig.index.intersection(old.index):
   a=sig.loc[t]; b=old.loc[t]; common=a.index.intersection(b.index); a=a[common];b=b[common]; ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1: vals.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if vals:
   mx=float(max(vals)); evidence.append((fid,mx,len(vals)));maxrho=max(maxrho,mx)
  else: missing.append(fid)
 except Exception: missing.append(fid)
print('LIBRARY_CORRELATION',json.dumps({'effective':len(effective),'audited':len(evidence),'max_abs_library_correlation':maxrho,'evidence':evidence,'missing':missing}))
