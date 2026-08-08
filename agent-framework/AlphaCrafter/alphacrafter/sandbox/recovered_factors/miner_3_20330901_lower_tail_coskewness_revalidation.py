"""Timely revalidation of the unadmitted lower-tail co-skewness candidate.
Uses completed daily bars through 2033-08-31.  The artifact scan is diagnostic only:
it deliberately cannot substitute for the mandatory all-admitted-factor audit.
"""
import glob, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-08-31')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
def tc(w):
 o=pd.DataFrame(index=r.index,columns=A,dtype=float); mp=max(15,int(.75*w))
 for a in A:
  peer=r.drop(columns=a).mean(axis=1)
  za=(r[a]-r[a].rolling(w,min_periods=mp).mean())/r[a].rolling(w,min_periods=mp).std()
  zp=(peer-peer.rolling(w,min_periods=mp).mean())/peer.rolling(w,min_periods=mp).std()
  o[a]=(za*zp.pow(2)).where(peer<0).rolling(w,min_periods=max(10,int(.45*w))).mean()
 return o
f=(-(tc(20)-tc(60))).replace([np.inf,-np.inf],np.nan)
print('FACTOR lower_tail_coskewness_contraction_20_60 VALIDATED_THROUGH',CUT.date())
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
results={}
for h in [1,3,5,7,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals)); ic=s.mean(); ir=ic/s.std(ddof=1)
 results[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,ic,ir,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   x=s.loc[lo:hi]; print('REGIME10',n,'dates',len(x),'IC=%+.6f ICIR=%+.6f hit=%.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 q=ranks.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
# Establish precisely whether persisted/artifact evidence covers all effective definitions.
effective=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE': effective.append(x['factor_id'])
 except Exception: pass
found=0; maxrho=-1; who=None
for fid in effective:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: continue
 found+=1; x=pd.read_pickle(max(hits,key=os.path.getmtime)); vals=[]
 for d in f.index.intersection(x.index):
  q=pd.concat([f.loc[d,f.columns.intersection(x.columns)].rename('a'),x.loc[d,f.columns.intersection(x.columns)].rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: vals.append(abs(spearmanr(q.a,q.b).statistic))
 if vals and max(vals)>maxrho: maxrho=max(vals); who=fid
print('INDEPENDENCE evidence_artifacts=%d effective_definitions=%d diagnostic_maxrho=%.6f factor=%s'%(found,len(effective),maxrho,who))
print('ADMISSION=FAIL: mandatory maximum correlation is unverified unless evidence_artifacts equals effective_definitions')
f.to_pickle('scripts/miner_3_20330901_lower_tail_coskewness_contraction_20_60_signal.pkl')
