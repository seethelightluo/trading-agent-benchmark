import os,json,pandas as pd,numpy as np
from scipy.stats import spearmanr
# One idea: conditional dollar-shock resilience.  The signal is negative 60d beta
# to DXY on days when SPX fell, rewarding assets that resist dollar-strength/risk-off shocks.
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-12-26')
def load(sym,root):
 for ext in ['.csv','.parquet']:
  p=f'{root}/{sym}{ext}'
  if os.path.exists(p):
   d=pd.read_csv(p,parse_dates=['date']) if ext=='.csv' else pd.read_parquet(p);return d[d.date<=CUT].set_index('date').sort_index().close
 raise FileNotFoundError(sym)
close=pd.DataFrame({a:load(a,'../persistent/stock_data') for a in A}).sort_index(); r=close.pct_change()
dxy=load('DXY','../persistent/index_data').reindex(close.index).pct_change(); spx=r.SPX
# covariance only conditional observations; 12 minimum avoids unstable ratios
cond=r.where(spx<0); dx=pd.DataFrame(np.tile(dxy.values,(len(A),1)).T;dx.index=r.index;dx.columns=A
n=(cond.notna()&dx.notna()).rolling(60).sum(); beta=cond.rolling(60,min_periods=12).cov(dxy).div(dxy.where(spx<0).rolling(60,min_periods=12).var(),axis=0)
sig=(-beta).where(n>=12).replace([np.inf,-np.inf],np.nan)
def stat(s,h):
 f=close.shift(-h).div(close).sub(1); z=[];ns=[]
 for t in s.index:
  m=s.loc[t].notna()&f.loc[t].notna()
  if m.sum()>=8:
   q=spearmanr(s.loc[t,m],f.loc[t,m]).statistic
   if np.isfinite(q):z.append(q);ns.append(m.sum())
 z=np.array(z);sd=z.std(ddof=1);return {'ic_dates':len(z),'mean_valid_instruments':float(np.mean(ns)),'ic':float(z.mean()),'icir':float(z.mean()/sd),'hit_ratio':float((z>0).mean()),'se':float(sd/np.sqrt(len(z)))}
print('IDEA conditional_dollar_shock_resilience_60obs');print('cutoff',CUT.date(),'panel_dates',len(sig),'coverage',round(float(sig.notna().mean().mean()),5))
for h in [1,5,10,20]:print('H',h,{k:round(v,6) for k,v in stat(sig,h).items()})
rr=[]
for i in range(1,len(sig)):
 m=sig.iloc[i].notna()&sig.iloc[i-1].notna()
 if m.sum()>=8:
  q=spearmanr(sig.iloc[i][m],sig.iloc[i-1][m]).statistic
  if np.isfinite(q):rr.append(q)
print('rank_stability',round(float(np.mean(rr)),6),'turnover_proxy',round(float(1-np.mean(rr)),6))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029','2029-01-01','2029-12-26')]:print('REG',nm,{k:round(v,6) for k,v in stat(sig.loc[lo:hi],5).items()})
# Admitted-factor artifact mapping, full library independence audit.
paths={}
for p in __import__('glob').glob('factors/*.json'):
 try:
  x=json.load(open(p));
  if x.get('validation',{}).get('status')=='EFFECTIVE':
   fid=x['factor_id']; hits=__import__('glob').glob('scripts/*'+fid+'*_signal.pkl')
   # known exceptional artifact aliases
   alias={'miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_2_standardized_jump_asymmetry_20v40obs':'scripts/miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl'}
   paths[fid]=alias.get(fid,hits[-1] if hits else None)
 except:pass
res=[]; missing=[]
for fid,p in paths.items():
 try:
  if not p:raise ValueError('no artifact')
  x=pd.read_pickle(p);x.index=pd.to_datetime(x.index);x=x.reindex(index=sig.index,columns=A); q=[]
  for t in sig.index:
   m=sig.loc[t].notna()&x.loc[t].notna()
   if m.sum()>=8:
    z=spearmanr(sig.loc[t,m],x.loc[t,m]).statistic
    if np.isfinite(z):q.append(z)
  if not q:raise ValueError('no finite correlation')
  res.append((fid,float(np.max(np.abs(q))),len(q)))
 except Exception as e:missing.append(fid)
res.sort(key=lambda z:-z[1]);print('AUDIT compared',len(res),'missing',missing);print('TOP_CORR',res[:8]);print('MAX_ABS_LIBRARY_CORR',res[0][1] if len(res)==len(paths) else 'UNAVAILABLE')
pd.to_pickle(sig,'scripts/miner_2_20291227_conditional_dollar_shock_resilience_60obs_signal.pkl')
