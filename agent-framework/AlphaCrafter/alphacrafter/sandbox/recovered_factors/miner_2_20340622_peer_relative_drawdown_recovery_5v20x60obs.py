# Peer-relative recovery-path candidate, single-idea validation
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-06-21')
# Daily synthetic worldline has common dates, but alignment is explicit.
cl=[]
for a in ASSETS:
    p='../persistent/stock_data/'+a+'.csv'
    x=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].rename(a)
    cl.append(x)
close=pd.concat(cl,axis=1).sort_index().loc[:CUTOFF]
r=close.pct_change()
# Factor: 5-session asset recovery net of a 60-session rolling beta to its peer basket,
# admitted only after an asset-specific 20-session drawdown (negative 20d return).
peer=(r.sum(axis=1,skipna=True).values[:,None]-r.values)/(r.notna().sum(axis=1).values[:,None]-1)
peer=pd.DataFrame(peer,index=r.index,columns=r.columns)
cov=r.rolling(60,min_periods=50).cov(peer) # pairwise same-column covariance
beta=cov.div(peer.rolling(60,min_periods=50).var(),axis=0)
raw=r.rolling(5,min_periods=5).sum()-beta*peer.rolling(5,min_periods=5).sum()
sig=raw.where(r.rolling(20,min_periods=20).sum()<0)

def ic_stats(h):
    fwd=close.shift(-h).div(close)-1
    rows=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
        if ok.sum()>=8:
            v=spearmanr(x[ok],y[ok]).statistic
            if np.isfinite(v): rows.append((dt,v,int(ok.sum())))
    z=pd.DataFrame(rows,columns=['date','ic','n'])
    if len(z)==0:return None,z
    mean=z.ic.mean(); sd=z.ic.std(ddof=1); ir=mean/sd if sd else np.nan
    return {'ic':mean,'icir':ir,'hit':(z.ic>0).mean(),'dates':len(z),'mean_n':z.n.mean()},z
print('candidate=peer_relative_drawdown_recovery_5v20x60obs')
print('source',close.index.min().date(),close.index.max().date(),'rows',len(close),'assets',len(ASSETS))
print('cell coverage',sig.notna().mean().mean(),'mean active',sig.notna().sum(axis=1).mean())
allz={}
for h in [1,5,10,20]:
    st,z=ic_stats(h); allz[h]=z
    print('horizon',h,st)
# Rank stability / turnover on consecutive valid signals.
ranks=sig.rank(axis=1,pct=True)
rs=[]
for i in range(1,len(ranks)):
 a,b=ranks.iloc[i-1],ranks.iloc[i]; ok=a.notna()&b.notna()
 if ok.sum()>=3: rs.append(spearmanr(a[ok],b[ok]).statistic)
print('rank_stability',np.nanmean(rs),'implied_turnover',1-np.nanmean(rs))
# Regimes on the strongest predeclared target 5d horizon.
z=allz[5]
for label,lo,hi in [('2026-2028','2026-01-01','2028-12-31'),('2029-2031','2029-01-01','2031-12-31'),('2032-2034','2032-01-01','2034-12-31')]:
 q=z[(z.date>=lo)&(z.date<=hi)]
 print('regime',label,'dates',len(q),'ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
# Correlation screen: only artifacts that actually contain aligned signals can be evidenced.
# Missing artifacts are deliberately reported and make admission impossible.
maxrho=0; evidence=[]; missing=[]
for f in glob.glob('factors/*.json'):
 d=json.load(open(f));
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 aid=d.get('factor_id',os.path.basename(f))
 art=d.get('signal_artifact')
 if not art or not os.path.exists(art): missing.append(aid); continue
 try:
  obj=pd.read_pickle(art)
  if isinstance(obj,pd.DataFrame): other=obj
  elif isinstance(obj,dict): other=pd.DataFrame(obj)
  else: raise ValueError('unsupported')
  # Expected date x asset artifact. Any other form is not complete evidence.
  other.index=pd.to_datetime(other.index)
  common_i=sig.index.intersection(other.index); common_c=sig.columns.intersection(other.columns)
  vals=[]
  for dt in common_i:
   a=sig.loc[dt,common_c]; b=other.loc[dt,common_c]; ok=a.notna()&b.notna()
   if ok.sum()>=8: vals.append(abs(spearmanr(a[ok],b[ok]).statistic))
  if vals:
   m=float(np.nanmax(vals)); evidence.append((aid,m)); maxrho=max(maxrho,m)
  else: missing.append(aid)
 except Exception: missing.append(aid)
print('library_correlation_evidenced',evidence)
print('max_abs_library_correlation_partial',maxrho)
print('library_evidence_complete',len(missing)==0,'missing_count',len(missing),'missing',missing)
print('ADMISSION: FAIL if missing evidence, irrespective of IC metrics')
