"""Miner_1: one candidate, EURUSD shock-conditioned volatility-normalized 10d reversal; cutoff 2035-07-04.
The factor is active only following a large EURUSD daily move and ranks assets by the inverse of
10-session return scaled by trailing 20-session realized volatility.  All thresholds are trailing-only.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-07-04')
def load(path):
    return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).loc[:END]
R=np.log(C).diff()
e=np.log(load('../persistent/index_data/EURUSD.csv').reindex(C.index).ffill()).diff()
# q=.85 is fixed before analysis; it selects exceptional, rather than routine, FX funding moves.
shock=(e.abs()>e.abs().rolling(60,min_periods=45).quantile(.85)).astype(float)
F=(-R.rolling(10,min_periods=10).sum()/R.rolling(20,min_periods=15).std().replace(0,np.nan)).mul(shock,axis=0)
def calc(h):
    y=C.shift(-h).div(C).sub(1); rec=[]; ns=[]
    for d in F.index:
        z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
        if len(z)>=8 and z.f.nunique()>1:
            v=spearmanr(z.f,z.y).statistic
            if np.isfinite(v): rec.append((d,v)); ns.append(len(z))
    x=pd.Series(dict(rec)); sd=x.std(ddof=1)
    return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd), 'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20,40]:
    _,m=calc(h); print('HORIZON',h,json.dumps(m,sort_keys=True))
x,_=calc(10)
for label,years in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_2033',[2031,2032,2033]),('2034_2035',[2034,2035])]:
    s=x[x.index.year.isin(years)]; print('REGIME_10D',label,'dates',len(s),'IC',float(s.mean()),'ICIR',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
    z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('PANEL',F.index.min().date(),END.date(),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'shock_dates',int(shock.sum()),'shock_frequency',float(shock.mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
# Contract audit: every admitted, non-backup JSON must have a matching signal artifact.
files=[p for p in glob.glob('factors/*.json') if not p.endswith('.bak')]
lib=[]; missing=[]
for p in files:
    try:
        j=json.load(open(p)); fid=j['factor_id']; hits=glob.glob('scripts/*'+fid+'*signal.pkl')
        if not hits: missing.append(fid); continue
        lib.append((fid,pd.read_pickle(hits[-1]).reindex(index=F.index,columns=A)))
    except Exception as ex: missing.append(os.path.basename(p)+':'+str(ex))
if missing: print('LIBRARY_AUDIT_FAILED',len(files),'files; missing',json.dumps(missing))
else:
    out=[]
    for fid,q in lib:
        z=[]
        for d in F.index:
            p=pd.concat([F.loc[d].rename('x'),q.loc[d].rename('y')],axis=1).dropna()
            if len(p)>=8 and p.x.nunique()>1 and p.y.nunique()>1:z.append(spearmanr(p.x,p.y).statistic)
        out.append((fid,float(np.mean(np.abs(z))),len(z)))
    print('LIBRARY_AUDIT_COMPLETE',len(out),'MAX_ABS',max(out,key=lambda v:v[1]),'ALL',json.dumps(out))
F.to_pickle('scripts/miner_1_20350705_eurusd_shock_volnorm_reversal_10v20obs_signal.pkl')
