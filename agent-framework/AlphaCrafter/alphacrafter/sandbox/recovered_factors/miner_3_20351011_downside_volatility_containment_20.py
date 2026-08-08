"""One idea: downside volatility containment (20d).
Higher signal denotes an asset whose trailing daily variability contains a smaller
share of negative-return variation, hypothesized to predict cross-asset returns.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-10-10')
def load(a):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
    return x.loc[:CUT,'close']
raw={a:load(a) for a in ASSETS}
end=min(x.dropna().index.max() for x in raw.values())
idx=pd.date_range(min(x.index.min() for x in raw.values()),end,freq='B')
close=pd.DataFrame(raw,index=idx).ffill()
r=np.log(close).diff().clip(-np.log(1.25),np.log(1.25))
# Negative semi-deviation divided by total RMS volatility; negate so containment ranks high.
down=np.sqrt(r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean())
total=np.sqrt(r.pow(2).rolling(20,min_periods=15).mean())
f=(1-down.div(total.replace(0,np.nan))).clip(0,1)
print('FACTOR downside_volatility_containment_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(ASSETS),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
    fw=close.shift(-h).div(close)-1; rows=[]; ns=[]
    for d in f.index:
        q=pd.concat((f.loc[d].rename('factor'),fw.loc[d].rename('forward')),axis=1).dropna()
        if len(q)>=8 and q.factor.nunique()>1 and q.forward.nunique()>1:
            v=spearmanr(q.factor,q.forward).statistic
            if np.isfinite(v): rows.append((d,v)); ns.append(len(q))
    s=pd.Series(dict(rows),dtype=float); ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
    s=ics[5].loc[lo:hi]
    print('REGIME5 %s dates=%d IC=%s ICIR=%s hit=%s'%(name,len(s),('%.6f'%s.mean() if len(s) else 'N/A'),('%.6f'%(s.mean()/s.std(ddof=1)) if len(s)>1 else 'N/A'),('%.6f'%(s>0).mean() if len(s) else 'N/A')))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
    q=ranks.iloc[[i-1,i]].T.dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(1),axis=0).div(f.std(1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turns),len(turns),z.abs().stack().mean()))
effective=[]
for fn in glob.glob('factors/*.json'):
    try:
        j=json.load(open(fn))
        if j.get('validation',{}).get('status')=='EFFECTIVE': effective.append(j['factor_id'])
    except Exception: pass
corr=[]; missing=[]
for fid in effective:
    ps=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not ps: missing.append(fid); continue
    old=pd.read_pickle(max(ps,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
    q=pd.concat([f.stack().rename('x'),old.stack().rename('y')],axis=1).dropna()
    if len(q)<8 or q.x.nunique()<2 or q.y.nunique()<2: missing.append(fid)
    else: corr.append(abs(spearmanr(q.x,q.y).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(effective),len(corr),len(missing),('%.6f'%max(corr) if len(corr)==len(effective) else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20351011_downside_volatility_containment_20_signal.pkl')
