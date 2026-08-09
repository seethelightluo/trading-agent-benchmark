"""One candidate: downside-risk-confirmed short reversal; no data after 2034-02-01."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-02-01')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[:CUT,'close']
p=pd.DataFrame({a:close(a) for a in A}); r=p.pct_change()
# Reversal strength is raised only when its trailing 20-session realized risk is
# dominated by losses (downside semivolatility / total rms return).
down=np.sqrt(r.clip(upper=0).pow(2).rolling(20,min_periods=12).mean())
tot=np.sqrt(r.pow(2).rolling(20,min_periods=12).mean())
f=(-p.pct_change(5)*down.div(tot.replace(0,np.nan))).replace([np.inf,-np.inf],np.nan)
print('FACTOR downside_semivol_confirmed_short_reversal_5_20d VALIDATED_THROUGH',CUT.date())
print('definition=negative trailing 5d return multiplied by trailing 20d downside semivolatility divided by total RMS return')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_34','2027-01-01',CUT)]:
 s=ics[1].loc[lo:hi];print('REGIME1 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(ts),len(ts)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn));
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else:print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340202_downside_semivol_confirmed_short_reversal_5_20d_signal.pkl')
