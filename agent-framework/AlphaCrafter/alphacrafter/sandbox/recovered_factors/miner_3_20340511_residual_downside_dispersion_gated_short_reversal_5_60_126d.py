"""One idea: idiosyncratic short-reversal gated by broad downside and high dispersion."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-05-10')
def close(a):
    return (pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])
            .set_index('date').sort_index()['close'])
p=pd.DataFrame({a:close(a) for a in A}).loc[:CUTOFF]
cut=p.dropna(how='all').index.max(); p=p.loc[:cut]
r1=p.pct_change(); r5=p.pct_change(5)
# Remove each asset's trailing 60-session loading on the equal-weight daily market return,
# then reverse only idiosyncratic 5d moves in a weak, high-dispersion tape.
mkt=r1.mean(axis=1)
beta=r1.rolling(60,min_periods=40).cov(mkt).unstack().reindex(columns=A).div(mkt.rolling(60,min_periods=40).var(),axis=0)
resid1=r1-beta.mul(mkt,axis=0)
resid5=resid1.rolling(5,min_periods=5).sum()
disp=r5.std(axis=1,ddof=1); pct=disp.rolling(126,min_periods=63).rank(pct=True)
state=(pct>=.75)&(r5.mean(axis=1)<0)
f=(-resid5).where(state,0.).replace([np.inf,-np.inf],np.nan)
print('FACTOR residual_downside_dispersion_gated_short_reversal_5_60_126d VALIDATED_THROUGH',cut.date())
print('definition=negative 5d sum of own daily returns residualized versus trailing-60d equal-weight-market beta, only if cross-asset 5d dispersion is top quartile of trailing 126d and equal-weight 5d return is negative')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d active_rate=%.6f' %(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),state.sum(),state.mean()))
ics={}
for h in [1,5,10,20]:
    fut=p.shift(-h).div(p)-1; out=[]; ns=[]
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fut.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            x=spearmanr(q.f,q.y).statistic
            if np.isfinite(x): out.append((d,x));ns.append(len(q))
    s=pd.Series(dict(out));ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f' %(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',cut)]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
r=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(r)):
 q=r.iloc[[i-1,i]].T.dropna()
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
 old=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),old.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else:print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340511_residual_downside_dispersion_gated_short_reversal_5_60_126d_signal.pkl')
