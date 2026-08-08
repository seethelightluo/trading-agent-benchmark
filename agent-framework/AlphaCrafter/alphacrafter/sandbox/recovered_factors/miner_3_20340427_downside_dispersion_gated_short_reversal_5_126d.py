"""One idea: downside-and-dispersion-gated five-day cross-asset reversal.
Research uses only bars no later than the supplied decision-date information cutoff.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-04-26') # prior completed day for 2034-04-27 decision

def close(a):
    return (pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])
            .set_index('date').sort_index()['close'])
p=pd.DataFrame({a:close(a) for a in A}).loc[:CUTOFF]
cut=p.dropna(how='all').index.max(); p=p.loc[:cut]
r1=p.pct_change(); r5=p.pct_change(5)
# State requires both a weak broad five-day tape and unusually dispersed opportunity set.
broad5=r5.mean(axis=1); dispersion=r5.std(axis=1,ddof=1)
disp_pct=dispersion.rolling(126,min_periods=63).rank(pct=True)
state=(disp_pct>=.75) & (broad5<0)
f=(-r5).where(state,0.0).replace([np.inf,-np.inf],np.nan)
print('FACTOR downside_dispersion_gated_short_reversal_5_126d VALIDATED_THROUGH',cut.date())
print('definition=-five-day own return only when five-day cross-asset dispersion is >= trailing-126d 75th percentile AND equal-weight five-day return is negative; zero otherwise')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d active_rate=%.6f' % (len(A),f.notna().any(axis=1).sum(),int(f.notna().sum().sum()),f.notna().mean().mean(),int(state.sum()),state.mean()))
ics={}
for h in [1,5,10,20]:
    future=p.shift(-h).div(p)-1
    out=[]; n=[]
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),future.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
            v=spearmanr(q.f,q.y).statistic
            if np.isfinite(v): out.append((d,v)); n.append(len(q))
    s=pd.Series(dict(out)); ics[h]=s
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f' % (h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(n)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',cut)]:
    s=ics[5].loc[lo:hi]
    print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f' % (nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
    q=ranks.iloc[[i-1,i]].T.dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
        turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d' % (np.mean(turns),len(turns)))
# Binding library test: only exact, historical signals count as evidence.
eff=[]
for fn in glob.glob('factors/*.json'):
    try:
        z=json.load(open(fn))
        if z.get('validation',{}).get('status')=='EFFECTIVE': eff.append(z['factor_id'])
    except Exception: pass
scores=[]; missing=[]
for fid in eff:
    hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not hits: missing.append(fid); continue
    old=pd.read_pickle(max(hits,key=os.path.getmtime))
    q=pd.concat([f.stack().rename('candidate'),old.stack().rename('library')],axis=1).dropna()
    if len(q)<8 or q.candidate.nunique()<2 or q.library.nunique()<2: missing.append(fid); continue
    scores.append((abs(spearmanr(q.candidate,q.library).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d' % (len(eff),len(scores),len(missing)))
if len(scores)==len(eff):
    v=max(scores); print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d' % v)
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340427_downside_dispersion_gated_short_reversal_5_126d_signal.pkl')
