"""One idea: commodity-versus-equity linkage differential.
Each asset is ranked by its trailing 60-day correlation with a commodity basket
(COPPER, WTI, XAU) minus its correlation with a global equity basket. The
signal measures a currently observable cross-asset linkage tilt, rather than
price direction or standalone volatility.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-04-25')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:CUT]
raw=pd.DataFrame({a:load(a) for a in A}); end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill(); r=p.pct_change()
commodity=r[['XAU','COPPER','WTI']].mean(axis=1)
equity=r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
# pairwise 60-completed-day correlation; own basket members are retained because
# their own linkage is a legitimate, directly measured cross-asset exposure.
fc=r.rolling(60,min_periods=45).corr(commodity)
fe=r.rolling(60,min_periods=45).corr(equity)
f=fc-fe
print('FACTOR commodity_equity_linkage_differential_60d VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
 s=ics[20].loc[lo:hi];print('REGIME20 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn));
  if j.get('validation',{}).get('status')=='EFFECTIVE':eff.append(j['factor_id'])
 except Exception:pass
scores=[];missing=[];peers=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths:missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime));old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),('%.6f'%max(scores) if scores and len(scores)==len(eff) else 'UNAVAILABLE')))
if peers:print('MAX_OBSERVED',max(peers))
f.to_pickle('scripts/miner_3_20350426_commodity_equity_linkage_differential_60d_signal.pkl')
