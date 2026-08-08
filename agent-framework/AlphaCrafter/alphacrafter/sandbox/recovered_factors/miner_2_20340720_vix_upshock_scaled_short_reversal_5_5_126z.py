"""One candidate: continuous VIX-upshock-scaled cross-asset short-term reversal."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def asset(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:asset(a) for a in A})
vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()
v=vx['close'] if 'close' in vx else vx.select_dtypes('number').iloc[:,0]
CUT=min(p.dropna(how='all').index.max(),v.index.max(),pd.Timestamp('2034-07-19'))
p=p.loc[:CUT]; v=v.reindex(p.index).ffill(); r=p.pct_change()
# A positive, fully lagged VIX 5d shock magnifies a contrarian 5d return signal.
# The continuous multiplier avoids selecting only a sparse set of shock dates.
shock=v.pct_change(5).shift(1)
scale=(shock-shock.rolling(126,min_periods=80).mean().shift(1))/shock.rolling(126,min_periods=80).std().shift(1)
scale=scale.clip(lower=0,upper=3)
f=(-r.rolling(5,min_periods=4).sum()).mul(scale,axis=0)
print('FACTOR vix_upshock_scaled_short_reversal_5_5_126z VALIDATED_THROUGH',CUT.date())
print('definition=negative trailing 5-session asset return multiplied by lagged positive standardized 5-session VIX change (126-session trailing mean/std; cap 3)')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_scale_dates=%d'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),(scale>0).sum()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; obs=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   c=spearmanr(q.f,q.y).statistic
   if np.isfinite(c):obs.append((d,c));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(ts),len(ts)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except Exception:pass
sc=[];missing=[]
for fid in eff:
 hit=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hit:missing.append(fid);continue
 z=pd.read_pickle(max(hit,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 sc.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(sc),len(missing)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(sc)[0] if len(sc)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20340720_vix_upshock_scaled_short_reversal_5_5_126z_signal.pkl')
