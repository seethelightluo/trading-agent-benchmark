"""One candidate: high-volatility-dispersion conditional residual reversal; leakage-safe labels."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-06-21')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close']
p=pd.DataFrame({a:close(a) for a in A}); r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
# Scores use only returns ending at t; forward labels are explicitly unavailable beyond CUT.
v20=r.rolling(20,min_periods=15).std().shift(1).replace(0,np.nan)
base=-resid.rolling(10,min_periods=8).sum().div(v20)
# Activate only when relative dispersion of lagged asset volatility is elevated versus its trailing 126 sessions.
disp=v20.std(axis=1).div(v20.mean(axis=1)).replace([np.inf,-np.inf],np.nan)
thresh=disp.rolling(126,min_periods=80).quantile(.60).shift(1)
f=base.where(disp>thresh, np.nan)
print('FACTOR high_vol_dispersion_conditional_residual_reversal_10_20_126q60 VALIDATED_THROUGH',CUT.date())
print('definition=negative 10-session cross-sectionally residualized return divided by lagged 20-session volatility, observed only when lagged cross-asset 20d-volatility dispersion exceeds its lagged 126d 60th percentile; higher scores are stressed-regime residual losers expected to reverse')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1
 # eliminate labels which need a price after the validation cutoff
 y.loc[y.index > CUT-pd.Timedelta(days=40)] = y.loc[y.index > CUT-pd.Timedelta(days=40)].where(y.index.to_series().apply(lambda d: d in p.index and (p.index.get_loc(d)+h)<len(p)).values[:,None])
 obs=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): obs.append((d,z)); ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[10].loc[lo:hi]; print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(ts),len(ts)))
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
 z=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2: missing.append(fid); continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(scores)[0] if len(scores)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20340622_high_vol_dispersion_conditional_residual_reversal_10_20_126q60_signal.pkl')
