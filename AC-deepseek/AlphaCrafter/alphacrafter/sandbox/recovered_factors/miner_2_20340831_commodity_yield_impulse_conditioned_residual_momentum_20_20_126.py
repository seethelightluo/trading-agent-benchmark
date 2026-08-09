"""One candidate: commodity-yield impulse conditioned residual momentum."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}); CUT=min(p.dropna(how='all').index.max(),pd.Timestamp('2034-08-30'));p=p.loc[:CUT];r=p.pct_change()
# Inflation-impulse regime: commodity basket's 20d return less the tradable US yield-series 20d return.
# A standardized, bounded signed regime multiplier turns residual 20d trend toward the prevailing macro impulse.
comm=r[['XAU','COPPER','WTI']].mean(axis=1)
imp=comm.rolling(20,min_periods=15).sum()-r['US10Y'].rolling(20,min_periods=15).sum()
z=(imp-imp.rolling(126,min_periods=80).mean().shift(1))/imp.rolling(126,min_periods=80).std().shift(1)
reg=np.tanh(z.shift(1))
res=r.sub(r.mean(axis=1),axis=0)
base=res.rolling(20,min_periods=15).sum().shift(1)
f=base.mul(reg,axis=0)
print('FACTOR commodity_yield_impulse_conditioned_residual_momentum_20_20_126 VALIDATED_THROUGH',CUT.date())
print('definition=completed 20-session asset return residual to equal-weight cross-asset return, multiplied by tanh of lagged 126-session z-score of (20-session equal-weight XAU/COPPER/WTI return minus US10Y return)')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f regime_pos=%.4f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),(reg>0).mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1;obs=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   c=spearmanr(q.f,q.y).statistic
   if np.isfinite(c):obs.append((d,c));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:tos.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(tos),len(tos)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn));
  if j.get('validation',{}).get('status')=='EFFECTIVE':eff.append(j['factor_id'])
 except:pass
sc=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 v=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),v.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 sc.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(sc),len(missing)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(sc)[0] if len(sc)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20340831_commodity_yield_impulse_conditioned_residual_momentum_20_20_126_signal.pkl')
