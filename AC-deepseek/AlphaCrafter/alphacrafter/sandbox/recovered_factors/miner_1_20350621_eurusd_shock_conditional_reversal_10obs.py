"""Miner_1 single-idea study: EURUSD shock-conditioned 10-session reversal; cutoff 2035-06-20."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-06-20')
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).loc[:END]
R=np.log(C).diff(); e=np.log(load('../persistent/index_data/EURUSD.csv').reindex(C.index).ffill()).diff()
# An extreme EURUSD move is a broad funding/FX shock. Within this state, rank assets by reversal of their preceding 10-day move.
shock=(e.abs()>e.abs().rolling(60,min_periods=45).quantile(.80)).astype(float)
F=-R.rolling(10,min_periods=10).sum().mul(shock,axis=0)
def calc(h):
 y=C.shift(-h).div(C).sub(1); out=[]; nn=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v):out.append((d,v));nn.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(nn))}
for h in [1,5,10,20,40]:
 _,m=calc(h); print('HORIZON',h,json.dumps(m,sort_keys=True))
x,_=calc(10)
for label, years in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_2033',[2031,2032,2033]),('2034_2035',[2034,2035])]:
 s=x[x.index.year.isin(years)]; print('REGIME_10D',label,'dates',len(s),'IC',float(s.mean()),'ICIR',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('PANEL',F.index.min().date(),END.date(),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'shock_dates',int(shock.sum()),'shock_frequency',float(shock.mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
# Mandatory full-library signal audit. Current admitted factor JSONs must each map to a saved signal artifact.
files=[p for p in glob.glob('factors/*.json') if not p.endswith('.bak')]
lib=[]; missing=[]
for p in files:
 try:
  j=json.load(open(p)); fid=j['factor_id']
  hits=glob.glob('scripts/*'+fid+'*signal.pkl')
  if not hits: missing.append(fid); continue
  q=pd.read_pickle(hits[-1]).reindex(index=F.index,columns=A)
  lib.append((fid,q))
 except Exception as ex: missing.append(os.path.basename(p)+':'+str(ex))
if missing: print('LIBRARY_AUDIT_FAILED',len(files),'files; missing',missing)
else:
 cs=[]
 for fid,q in lib:
  vals=[]
  for d in F.index:
   z=pd.concat([F.loc[d].rename('x'),q.loc[d].rename('y')],axis=1).dropna()
   if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.x,z.y).statistic)
  cs.append((fid,float(np.nanmean(np.abs(vals))),len(vals)))
 print('LIBRARY_AUDIT_COMPLETE',len(cs),'MAX_ABS',max(cs,key=lambda x:x[1]),'ALL',json.dumps(cs))
F.to_pickle('scripts/miner_1_20350621_eurusd_shock_conditional_reversal_10obs_signal.pkl')
