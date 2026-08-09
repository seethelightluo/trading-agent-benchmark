"""Miner 2: directional intraday range asymmetry, one interpretable path-shape idea."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2030-01-23') # prior completed day for 2030-01-24 decision
root='../persistent/stock_data'
def col(a,c):
 return pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date')[c].sort_index().loc[:CUT]
close=pd.DataFrame({a:col(a,'close') for a in A}); op=pd.DataFrame({a:col(a,'open') for a in A}); hi=pd.DataFrame({a:col(a,'high') for a in A}); lo=pd.DataFrame({a:col(a,'low') for a in A})
# Mean signed close-location in its daily range.  Negative sign tests short-run mean reversion after persistent close-at-high behavior.
rng=(hi-lo).replace(0,np.nan); loc=(close-op).div(rng).clip(-1,1)
sig=(-loc.rolling(10,min_periods=7).mean()).replace([np.inf,-np.inf],np.nan)
def stats(s,h):
 y=close.shift(-h).div(close)-1; out=[]
 for t in s.index:
  ok=s.loc[t].notna()&y.loc[t].notna()
  if ok.sum()>=8:
   v=spearmanr(s.loc[t,ok],y.loc[t,ok]).statistic
   if np.isfinite(v):out.append((t,v,ok.sum()))
 z=pd.DataFrame(out,columns=['date','ic','n']); sd=z.ic.std(ddof=1)
 return z,{'ic_dates':len(z),'mean_valid_instruments':z.n.mean(),'ic':z.ic.mean(),'icir':z.ic.mean()/sd if sd else np.nan,'hit_ratio':(z.ic>0).mean(),'se':sd/np.sqrt(len(z)) if len(z) else np.nan}
print('IDEA inverse_persistent_intraday_range_location_10obs','cutoff',CUT.date(),'panel_dates',len(sig),'assets',len(A))
print('COVERAGE',sig.notna().mean().mean(),'MEAN_VALID',sig.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 z,x=stats(sig,h);print('H',h,json.dumps(x,default=float))
 for nm,aa,bb in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_30','2029-01-01','2030-01-23')]:
  q=z[(z.date>=aa)&(z.date<=bb)];sd=q.ic.std(ddof=1);print('REGIME',h,nm,'dates',len(q),'ic',q.ic.mean(),'icir',q.ic.mean()/sd if sd else np.nan,'hit',(q.ic>0).mean())
st=[]
for i in range(1,len(sig)):
 ok=sig.iloc[i].notna()&sig.iloc[i-1].notna()
 if ok.sum()>=8:st.append(spearmanr(sig.iloc[i,ok],sig.iloc[i-1,ok]).statistic)
print('TURNOVER rank_stability',np.mean(st),'implied_daily',(1-np.mean(st))/2)
# Binding maximum absolute daily cross-sectional Spearman correlation against every admitted panel.
eff=[]
for p in glob.glob('factors/*.json'):
 try:
  d=json.load(open(p));
  if d.get('validation',{}).get('status')=='EFFECTIVE':eff.append(d['factor_id'])
 except Exception:pass
found=[]; missing=[]; best=(-1,None)
for fid in eff:
 hits=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not hits: missing.append(fid);continue
 try:
  g=pd.read_pickle(hits[-1]);g.index=pd.to_datetime(g.index);g=g.reindex(index=sig.index,columns=A); vals=[]
  for t in sig.index:
   ok=sig.loc[t].notna()&g.loc[t].notna()
   if ok.sum()>=8:
    v=spearmanr(sig.loc[t,ok],g.loc[t,ok]).statistic
    if np.isfinite(v):vals.append(abs(v))
  if not vals: raise ValueError('no overlap')
  mx=max(vals);found.append(fid)
  if mx>best[0]:best=(mx,fid)
 except Exception:missing.append(fid)
print('AUDIT effective',len(eff),'resolved',len(found),'missing',missing,'max_abs_library_correlation',best[0] if not missing else 'UNAVAILABLE','observed_max',best[0],'most_correlated',best[1])
sig.to_pickle('scripts/miner_2_20300124_inverse_persistent_intraday_range_location_10obs_signal.pkl')
