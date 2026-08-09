"""One candidate: VIX-shock conditional cross-asset defensive response momentum."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-07-05')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load_asset(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close']
def load_vix():
 x=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()
 col='close' if 'close' in x else x.select_dtypes('number').columns[0]
 return x.loc[:CUT,col]
p=pd.DataFrame({a:load_asset(a) for a in A}); r=p.pct_change(); v=load_vix().reindex(p.index).ffill()
# On a lagged VIX shock, rank assets by their preceding 10-day response. Positive values
# favor instruments already acting defensively; no macro series is ever a tradable column.
vshock=v.pct_change(5).shift(1)
threshold=vshock.rolling(126,min_periods=80).quantile(.60).shift(1)
active=vshock>threshold
f=r.rolling(10,min_periods=8).sum().where(active, np.nan)
print('FACTOR vix_shock_conditional_defensive_response_momentum_10_5_126q60 VALIDATED_THROUGH',CUT.date())
print('definition=trailing 10-session asset return, active only if lagged 5-session VIX change exceeds its lagged 126-session 60th percentile')
print('assets=%d active_dates=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),active.sum(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; obs=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): obs.append((d,z));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[10].loc[lo:hi]; print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
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
f.to_pickle('scripts/miner_2_20340706_vix_shock_conditional_defensive_response_momentum_10_5_126q60_signal.pkl')
