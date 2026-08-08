"""One idea: cross-asset momentum acceleration (20d return minus 60d return)."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-01-03')
def load(a):
    return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:CUT].reindex(pd.date_range('2020-01-01',CUT,freq='B')).ffill()
# Higher score means recent 20-session performance is stronger than its own medium-term 60-session trend.
f=p.pct_change(20)-p.pct_change(60)
print('FACTOR momentum_acceleration_20_60d VALIDATED_THROUGH',CUT.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
    vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            v=spearmanr(q.f,q.y).statistic
            if np.isfinite(v): vals.append((d,v)); ns.append(len(q))
    s=pd.Series(dict(vals)); ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01','2033-12-31'),('2034_2035','2034-01-01',CUT)]:
    s=ics[10].loc[lo:hi]
    print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
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
sc=[]; miss=[]
for fid in eff:
    h=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not h: miss.append(fid); continue
    old=pd.read_pickle(max(h,key=os.path.getmtime))
    if isinstance(old,dict): old=old.get('signal',old.get('factor'))
    if not isinstance(old,pd.DataFrame): miss.append(fid); continue
    q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
    if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: miss.append(fid)
    else: sc.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(sc),len(miss),('%.6f'%max(sc) if len(sc)==len(eff) and sc else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_2_20350104_momentum_acceleration_20_60d_signal.pkl')
