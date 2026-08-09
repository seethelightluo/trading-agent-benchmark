"""One idea: upside-minus-downside beta asymmetry (60d) to cross-asset median returns."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2034-12-06')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:CUT].reindex(pd.date_range('2020-01-01',CUT,freq='B')).ffill(); r=p.pct_change(); m=r.median(axis=1)
# 60d conditional beta asymmetry. Each state needs 15 observations (rather than 30) because the two mutually exclusive states split the window.
pos=m.where(m>0); neg=m.where(m<0)
bp=r.mul(pos,axis=0).rolling(60,min_periods=15).sum().div(pos.pow(2).rolling(60,min_periods=15).sum().replace(0,np.nan),axis=0)
bn=r.mul(neg,axis=0).rolling(60,min_periods=15).sum().div(neg.pow(2).rolling(60,min_periods=15).sum().replace(0,np.nan),axis=0)
f=bp-bn
print('FACTOR upside_downside_beta_asymmetry_60d VALIDATED_THROUGH',CUT.date()); print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s; print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',CUT)]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
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
sc=[];miss=[]
for fid in eff:
 h=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not h:miss.append(fid);continue
 old=pd.read_pickle(max(h,key=os.path.getmtime));q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:miss.append(fid)
 else:sc.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(sc),len(miss),('%.6f'%max(sc) if len(sc)==len(eff) and sc else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20341207_upside_downside_beta_asymmetry_60d_signal.pkl')
