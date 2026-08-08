"""One candidate: high-dispersion residual pullback reversal after a short bounce.
Tests whether medium-term residual losers mean revert only when cross-asset
return dispersion is elevated and their most recent residual return has bounced.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-04-26')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close']
p=pd.DataFrame({a:load(a) for a in A}); r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
# t uses only completed prices through t; lagged vol prevents a contemporaneous scale choice.
med=resid.rolling(40,min_periods=40).sum()-resid.rolling(5,min_periods=5).sum()
bounce=resid.rolling(5,min_periods=5).sum()
vol=r.rolling(20,min_periods=15).std().shift(1).replace(0,np.nan)
disp=resid.std(axis=1); gate=disp.gt(disp.rolling(60,min_periods=60).quantile(.60))
# Negative medium residual trend, activated only after a residual bounce in elevated dispersion.
f=(-med.div(vol)).where(gate, np.nan).where(bounce.gt(0),np.nan)
print('FACTOR high_dispersion_residual_pullback_bounce_reversal_40_5_vol20_q60 VALIDATED_THROUGH',CUT.date())
print('definition=negative 40d-minus-last5d residual trend / lagged 20d vol, active only when 5d residual return is positive and daily residual dispersion exceeds rolling-60d 60th percentile')
print('assets=%d factor_dates=%d cells=%d coverage=%.4f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; obs=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):obs.append((d,v));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[10].loc[lo:hi];print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]; scores=[]; missing=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except:pass
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(scores)[0] if len(scores)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20340427_high_dispersion_residual_pullback_bounce_reversal_40_5_vol20_q60_signal.pkl')
