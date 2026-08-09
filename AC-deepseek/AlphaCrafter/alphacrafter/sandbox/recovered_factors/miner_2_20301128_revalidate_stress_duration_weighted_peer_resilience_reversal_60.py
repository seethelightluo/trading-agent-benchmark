"""Scheduled revalidation of one admitted factor: stress-duration peer resilience reversal."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def close(a):
 d=get_stock_daily_data(a,5000).copy().sort_values('date'); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.to_numeric(d.drop_duplicates('date',keep='last').set_index('date')['close'],errors='coerce')
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=P.pct_change(); M=R.median(axis=1); REL=R.sub(M,axis=0)
stress=M < -.35*M.rolling(60,min_periods=30).std()
conditional=REL.where(np.broadcast_to(stress.to_numpy()[:,None],REL.shape))
weighted=conditional.mul(1+.25*stress.rolling(5,min_periods=1).sum().shift(1),axis=0)
F=-weighted.rolling(60,min_periods=5).mean(); F=F.sub(F.median(axis=1),axis=0).shift(1)
print('FACTOR stress_duration_weighted_peer_resilience_reversal_60 ENDPOINT',P.index.max().date(),'ASSETS',len(A),'ROWS',len(P))
out={}
for h in [1,5,10,20]:
 vals=[]
 for t in F.index:
  z=pd.concat([F.loc[t],P.shift(-h).loc[t]/P.loc[t]-1],axis=1).dropna()
  if len(z)>=8:
   rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(rho):vals.append((t,rho,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']); out[h]=q
 ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print(f'H{h} IC {ic:.6f} ICIR {ir:.6f} HIT {(q.ic>0).mean():.4f} DATES {len(q)} NMEAN {q.n.mean():.3f} NMIN {q.n.min()}')
 for nm,mask in [('2020_2022',q.date<'2023-01-01'),('2023_2025',(q.date>='2023-01-01')&(q.date<'2026-01-01')),('2026_2028',(q.date>='2026-01-01')&(q.date<'2029-01-01')),('2029_current',q.date>='2029-01-01'),('recent180',q.date>=q.date.max()-pd.Timedelta(days=180))]:
  v=q.loc[mask,'ic']; print(' REG',nm,'DATES',len(v),'IC',round(v.mean(),6) if len(v) else None,'ICIR',round(v.mean()/v.std(ddof=1),6) if len(v)>1 else None,'HIT',round((v>0).mean(),4) if len(v) else None)
print('COVERAGE',int(F.notna().sum().sum()),'/',F.size,round(F.notna().mean().mean(),6),'TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'CROSS_SD',round(F.std(axis=1).mean(),6),'STRESS_SHARE',round(stress.mean(),6))
q=out[20]; print('GATE20',abs(q.ic.mean())>=.007 and abs(q.ic.mean()/q.ic.std(ddof=1))>=.084)
