"""One idea: inverse-volatility 5d reversal activated after broad cross-asset drawdowns."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2034-09-27')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:CUT]
p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='B')).ffill(); r=p.pct_change()
# A broad drawdown is an observable, asset-neutral state: median 20d cross-asset return is in its worst trailing-year quartile.
breadth_ret=p.pct_change(20).median(axis=1)
state=breadth_ret < breadth_ret.rolling(252,min_periods=126).quantile(.25)
f=(-p.pct_change(5).div(r.rolling(20,min_periods=15).std().replace(0,np.nan))).where(state)
print('FACTOR broad_drawdown_inverse_vol_reversal_5_20_252 VALIDATED_THROUGH',p.dropna(how='all').index.max().date())
print('definition=-(5d own return)/20d own return volatility, emitted only when median 20d return across tradable assets is below its trailing-252d 25th percentile')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),state.sum()))
ics={}
for h in [1,5,10,20]:
 out=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): out.append((d,v)); ns.append(len(q))
 s=pd.Series(dict(out),dtype=float); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',CUT)]:
 s=ics[5].loc[lo:hi]; print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE': eff.append(z['factor_id'])
 except Exception: pass
scores=[]; missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid); continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid)
 else: scores.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(eff) and scores else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20340928_broad_drawdown_inverse_vol_reversal_5_20_252_signal.pkl')
