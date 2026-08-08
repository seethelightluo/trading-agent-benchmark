"""One idea: VIX-rise conditioned inverse residual tail-asymmetry, 20d/60d."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-08-02')
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:close(a) for a in A}).loc[:CUTOFF]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:CUTOFF]
cut=min(p.dropna(how='all').index.max(),vix.dropna().index.max());p=p.loc[:cut];vix=vix.loc[:cut]
r=p.pct_change(); market=r.mean(axis=1); var=market.rolling(60,min_periods=40).var().replace(0,np.nan)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(market).div(var) for a in A})
e=r-beta.mul(market,axis=0)
# Inverse recent tail asymmetry: assets whose downside residual semivariance has risen relative to upside are preferred after a VIX rise.
down=e.clip(upper=0).pow(2).rolling(20,min_periods=15).mean(); up=e.clip(lower=0).pow(2).rolling(20,min_periods=15).mean()
base_down=e.clip(upper=0).pow(2).rolling(60,min_periods=40).mean(); base_up=e.clip(lower=0).pow(2).rolling(60,min_periods=40).mean()
raw=-((down/(up+1e-12))-(base_down/(base_up+1e-12)))
state=vix.pct_change(5)>0
f=raw.where(state, np.nan).replace([np.inf,-np.inf],np.nan)
print('FACTOR vix_rise_inverse_residual_tail_asymmetry_20_60d VALIDATED_THROUGH',cut.date())
print('definition=-(recent20 residual downside/upside semivariance ratio - trailing60 ratio), observable only when VIX 5d return > 0')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f state_fraction=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),state.mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; out=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):out.append((d,z));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',cut)]:
 s=ics[5].loc[lo:hi]; print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn));
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except: pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid);continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),old.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff): print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340803_vix_rise_inverse_residual_tail_asymmetry_20_60d_signal.pkl')
