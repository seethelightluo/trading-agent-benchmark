"""Miner 1: conditional downside peer-beta contraction (20d vs 60d).
Higher score means an asset's loading on broad-peer losses has fallen recently versus baseline.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-09-14')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
def downside_beta(w):
 out=pd.DataFrame(index=r.index,columns=A,dtype=float)
 for a in A:
  peer=r.drop(columns=a).mean(axis=1)
  # rolling covariance conditional on peer-return being negative; require 60% of expected negative observations
  mask=peer<0
  x=peer.where(mask); y=r[a].where(mask)
  n=mask.rolling(w).sum(); cov=y.rolling(w,min_periods=1).cov(x); var=x.rolling(w,min_periods=1).var()
  out[a]=(cov/var).where(n>=int(w*.27))
 return out
# positive = lower recent downside loading / better loss resilience
f=-(downside_beta(20)-downside_beta(60)).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE conditional_downside_peer_beta_contraction_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',f.dropna(how='all').shape[0],'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; rows=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):rows.append((d,z));ns.append(len(q))
 s=pd.Series(dict(rows),dtype=float);ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi]; print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
f.to_pickle('scripts/miner_1_20330915_conditional_downside_peer_beta_contraction_20_60_signal.pkl')
print('INDEPENDENCE not evaluated: complete current-library signal artifacts are not resolvable by factor_id; admission requires complete evidence.')
