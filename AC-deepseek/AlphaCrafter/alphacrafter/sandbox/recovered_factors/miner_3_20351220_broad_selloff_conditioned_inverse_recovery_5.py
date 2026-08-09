"""One idea: broad-selloff-conditioned inverse recovery.
A cross-asset short-horizon reversal signal, activated smoothly only when the
median universe has sold off. This separates ordinary recovery momentum from
liquidity/risk-off overshoots, using completed closes only.
"""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-12-19')
def get(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close']
r={a:get(a) for a in A}; end=min(x.dropna().index.max() for x in r.values())
ix=pd.date_range(min(x.index.min() for x in r.values()),end,freq='B'); c=pd.DataFrame(r,index=ix).ffill()
ret5=c.pct_change(5).clip(-.75,.75); med=ret5.median(axis=1)
# Smooth 0--1 activation grows as the cross-asset five-day median falls below -1%.
state=((-.01-med)/.04).clip(0,1)
# Under broad selloffs, favor the most oversold assets; otherwise retain a small
# reversal component rather than a sparse binary signal.
f= -ret5.mul(.25+.75*state,axis=0)
print('FACTOR broad_selloff_conditioned_inverse_recovery_5 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f mean_state=%.6f active_state_days=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),state.mean(),(state>0).mean()))
ics={}
for h in [1,5,10,20]:
 fw=c.shift(-h).div(c)-1; out=[]; nn=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1:
   v=spearmanr(q.f,q.r).statistic
   if np.isfinite(v):out.append((d,v));nn.append(len(q))
 s=pd.Series(dict(out));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(nn)))
for n,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
 s=ics[5].loc[lo:hi]; print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn));
  if j.get('validation',{}).get('status')=='EFFECTIVE':eff.append(j['factor_id'])
 except:pass
co=[]; miss=[]
for fid in eff:
 p=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not p:miss.append(fid);continue
 o=pd.read_pickle(max(p,key=os.path.getmtime)); o=o.get('signal',o) if isinstance(o,dict) else o
 q=pd.concat([f.stack().rename('x'),o.stack().rename('y')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.y.nunique()<2:miss.append(fid)
 else:co.append(abs(spearmanr(q.x,q.y).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(co),len(miss),('%.6f'%max(co) if len(co)==len(eff) else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20351220_broad_selloff_conditioned_inverse_recovery_5_signal.pkl')
