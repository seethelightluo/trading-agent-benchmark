import os, glob, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
# One idea: VIX-upside beta defensiveness. Estimate each asset's 40-session
# sensitivity only on VIX-up days; a low beta signals relative shock resilience.
END='2029-06-27'
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END] for a in ASSETS}
close=pd.DataFrame({a:x['close'] for a,x in raw.items()}).sort_index()
ret=close.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
vc=vix['close'] if 'close' in vix else vix.iloc[:,0]
vr=vc.pct_change().reindex(close.index)
# beta on positive VIX-return sessions, requiring 10 observations; negate beta.
sig=pd.DataFrame(index=close.index,columns=ASSETS,dtype=float)
for i,t in enumerate(close.index):
    if i<40: continue
    ix=close.index[i-39:i+1]; x=vr.reindex(ix); mask=x>0
    if mask.sum()<10: continue
    xv=x[mask]; var=xv.var(ddof=1)
    if not np.isfinite(var) or var==0: continue
    for a in ASSETS:
        y=ret.loc[ix,a][mask]; ok=y.notna()&xv.notna()
        if ok.sum()>=10: sig.loc[t,a]=-y[ok].cov(xv[ok])/xv[ok].var(ddof=1)
def calc(s,h,mask=None):
 fwd=close.shift(-h)/close-1; vals=[];ns=[]
 dates=s.index if mask is None else s.index[np.asarray(mask,dtype=bool)]
 for t in dates:
  x=s.loc[t];y=fwd.loc[t];ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1 and y[ok].nunique()>1: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
 z=np.array(vals);sd=z.std(ddof=1) if len(z)>1 else np.nan
 return {'ic':float(z.mean()) if len(z) else None,'icir':float(z.mean()/sd) if sd>0 else None,'hit_ratio':float((z>0).mean()) if len(z) else None,'ic_dates':len(z),'mean_valid_instruments':float(np.mean(ns)) if ns else 0}
print('idea=negative beta to positive VIX moves 40obs; endpoint='+END)
print('PANEL',json.dumps({'dates':len(sig),'coverage':float(sig.notna().mean().mean()),'mean_valid_instruments':float(sig.notna().sum(1).mean()),'latest_valid':str(sig.dropna(how="all").index.max().date())}))
for h in [1,5,10,20]:print('HORIZON',h,json.dumps(calc(sig,h)))
for label,mask in [('2020_2022',sig.index<='2022-12-31'),('2023_2025',(sig.index>='2023-01-01')&(sig.index<='2025-12-31')),('2026_2028',(sig.index>='2026-01-01')&(sig.index<='2028-12-31')),('2029_ytd',sig.index>='2029-01-01')]:print('REGIME',label,json.dumps(calc(sig,1,mask)))
rs=[]
for i in range(1,len(sig)):
 a,b=sig.iloc[i-1],sig.iloc[i];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:rs.append(spearmanr(a[ok],b[ok]).statistic)
print('TURNOVER',json.dumps({'daily_rank_stability':float(np.mean(rs)),'implied_daily_rank_turnover':float(1-np.mean(rs)),'pairs':len(rs)}))
out='scripts/miner_2_20290628_vix_up_beta_defensiveness_40obs_signal.pkl';sig.to_pickle(out)
effective=[]
for f in glob.glob('factors/*.json'):
 try:
  j=json.load(open(f));
  if not f.endswith('.bak') and j.get('validation',{}).get('status')=='EFFECTIVE': effective.append(j['factor_id'])
 except: pass
maxrho=0.;ev=[];missing=[]
for fid in effective:
 paths=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not paths: missing.append(fid);continue
 try:
  old=pd.read_pickle(sorted(paths)[-1]);old.index=pd.to_datetime(old.index);vs=[]
  for t in sig.index.intersection(old.index):
   a=sig.loc[t,sig.columns.intersection(old.columns)];b=old.loc[t,a.index];ok=a.notna()&b.notna()
   if ok.sum()>=8 and a[ok].nunique()>1 and b[ok].nunique()>1:vs.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if vs: ev.append((fid,float(max(vs)),len(vs)));maxrho=max(maxrho,max(vs))
  else: missing.append(fid)
 except: missing.append(fid)
print('LIBRARY_CORRELATION',json.dumps({'effective':len(effective),'audited':len(ev),'max_abs_library_correlation':maxrho,'evidence':ev,'missing':missing}))
