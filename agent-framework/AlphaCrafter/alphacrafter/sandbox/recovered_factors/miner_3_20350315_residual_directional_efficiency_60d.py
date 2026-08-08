"""One idea: residual directional efficiency, removing conventional return and volatility.
At each date, cross-sectionally regress 60-day signed directional efficiency on
60-day return and 60-day realized volatility (with intercept); signal is the
residual. This isolates path persistence/choppiness rather than trend strength
or risk level. All inputs are completed closes at t."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-03-14')
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:CUT]
raw=pd.DataFrame({a:load(a) for a in A}); end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill(); r=p.pct_change()
ret60=p.pct_change(60); vol60=r.rolling(60,min_periods=60).std()
eff=ret60.div(r.abs().rolling(60,min_periods=60).sum())
# OLS residual; requires >= 8 assets, giving 5+ residual degrees of freedom.
f=pd.DataFrame(np.nan,index=p.index,columns=A)
for d in p.index:
    q=pd.concat([eff.loc[d].rename('eff'),ret60.loc[d].rename('ret'),vol60.loc[d].rename('vol')],axis=1).dropna()
    if len(q)>=8 and q[['ret','vol']].nunique().min()>1:
        x=np.column_stack([np.ones(len(q)),q.ret.values,q.vol.values])
        f.loc[d,q.index]=q.eff.values-x@np.linalg.lstsq(x,q.eff.values,rcond=None)[0]
print('FACTOR residual_directional_efficiency_60d VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
    vals=[];ns=[];fw=p.shift(-h).div(p)-1
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            v=spearmanr(q.f,q.y).statistic
            if np.isfinite(v):vals.append((d,v));ns.append(len(q))
    s=pd.Series(dict(vals));ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
    s=ics[20].loc[lo:hi]; print('REGIME20 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
    q=rk.iloc[[i-1,i]].T.dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
effids=[]
for fn in glob.glob('factors/*.json'):
    try:
        j=json.load(open(fn))
        if j.get('validation',{}).get('status')=='EFFECTIVE':effids.append(j['factor_id'])
    except Exception:pass
scores=[];missing=[];peers=[]
for fid in effids:
    paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not paths:missing.append(fid);continue
    old=pd.read_pickle(max(paths,key=os.path.getmtime));old=old.get('signal',old) if isinstance(old,dict) else old
    q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
    if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
    rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(effids),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(effids) and scores else 'UNAVAILABLE')))
if peers:print('MAX_OBSERVED',max(peers))
f.to_pickle('scripts/miner_3_20350315_residual_directional_efficiency_60d_signal.pkl')
