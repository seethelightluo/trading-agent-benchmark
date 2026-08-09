"""One idea: acceleration of upside/downside beta asymmetry, 20d versus 60d."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2034-12-20')
def load(a):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
    return x.loc[:CUT]
raw=pd.DataFrame({a:load(a) for a in A})
# Stop at last date for which the complete cross-asset panel is genuinely observed; no forward filling beyond data availability.
end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill()
r=p.pct_change(); market=r.median(axis=1)
def asym(w):
    pos=market.where(market>0); neg=market.where(market<0)
    bp=r.mul(pos,axis=0).rolling(w,min_periods=max(8,w//4)).sum().div(pos.pow(2).rolling(w,min_periods=max(8,w//4)).sum().replace(0,np.nan),axis=0)
    bn=r.mul(neg,axis=0).rolling(w,min_periods=max(8,w//4)).sum().div(neg.pow(2).rolling(w,min_periods=max(8,w//4)).sum().replace(0,np.nan),axis=0)
    return bp-bn
f=asym(20)-asym(60)
print('FACTOR conditional_beta_asymmetry_acceleration_20_60 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
    vals=[];ns=[]; fw=p.shift(-h).div(p)-1
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            z=spearmanr(q.f,q.y).statistic
            if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
    s=pd.Series(dict(vals));ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',end)]:
 s=ics[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
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
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid); continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid)
 else: scores.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(eff) and scores else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20341221_conditional_beta_asymmetry_acceleration_20_60_signal.pkl')
