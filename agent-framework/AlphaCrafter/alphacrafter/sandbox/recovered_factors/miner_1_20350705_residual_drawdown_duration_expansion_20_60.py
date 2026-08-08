"""One price-only factor: residual drawdown-duration expansion, 20/60d.
The signal is time spent below the rolling 60d high (capped at 20 days),
residualized each day against current 60d drawdown, 20d return and 20d
realized volatility. It tests whether persistent failure to recover contains
cross-asset information beyond severity, recent direction, and risk."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
raw=pd.DataFrame({a:load(a) for a in A})
end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill()
r=p.pct_change(); high=p.rolling(60,min_periods=60).max(); dd=p.div(high)-1
# Days since most recent rolling-window high; events older than 20 sessions are equally persistent.
dur=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
    hit=(p[a]>=high[a]*(1-1e-12)).to_numpy()
    out=np.full(len(p),np.nan); last=-10**6
    for i,v in enumerate(hit):
        if v: last=i
        if i>=59: out[i]=min(i-last,20)
    dur[a]=out
ret20=p.pct_change(20); vol20=r.rolling(20,min_periods=20).std()
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for d in p.index:
    q=pd.DataFrame({'y':dur.loc[d],'dd':dd.loc[d],'ret':ret20.loc[d],'vol':vol20.loc[d]}).dropna()
    if len(q)>=8:
        X=np.c_[np.ones(len(q)),q[['dd','ret','vol']].to_numpy()]
        f.loc[d,q.index]=q.y-X.dot(np.linalg.lstsq(X,q.y,rcond=None)[0])
print('FACTOR residual_drawdown_duration_expansion_20_60 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in (1,5,10,20):
    fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            x=spearmanr(q.f,q.y).statistic
            if np.isfinite(x): vals.append((d,x)); ns.append(len(q))
    s=pd.Series(dict(vals));ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for label,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_29','2025-01-01','2029-12-31'),('2030_34','2030-01-01','2034-12-31'),('2035','2035-01-01',str(end.date()))]:
    s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
    q=rk.iloc[[i-1,i]].T.dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turns),len(turns),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
    try:
        j=json.load(open(fn))
        if j.get('validation',{}).get('status')=='EFFECTIVE': eff.append(j['factor_id'])
    except Exception: pass
scores=[]; missing=[]; peers=[]
for fid in eff:
    paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not paths: missing.append(fid);continue
    old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
    q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
    if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid);continue
    rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
mx=max(scores) if len(scores)==len(eff) and scores else None
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),'%.6f'%mx if mx is not None else 'UNAVAILABLE'))
if peers: print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
f.to_pickle('scripts/miner_1_20350705_residual_drawdown_duration_expansion_20_60_signal.pkl')
