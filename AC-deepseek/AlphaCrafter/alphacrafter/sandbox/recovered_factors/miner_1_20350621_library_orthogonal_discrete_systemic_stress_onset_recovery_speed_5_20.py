"""One candidate: library-orthogonal discrete systemic-stress onset recovery speed (5/20)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']; r=P.pct_change()
# Identify new broad stress episodes: peer return below its trailing 60d 20th percentile,
# after a non-stress day. Rank assets by peer-relative rebound over the next five observed
# sessions, scaled by each asset's own recent volatility.
peer=r.median(axis=1); threshold=peer.rolling(60,min_periods=40).quantile(.20)
stress=(peer<threshold)&~(peer.shift(1)<threshold.shift(1))
rel=r.sub(peer,axis=0); ownvol=r.rolling(20,min_periods=12).std().replace(0,np.nan)
# trailing observed outcome: returns from onset through sessions t-4, available with lag.
onset_rel=rel.where(stress, np.nan).ffill(limit=4)
age=pd.Series(np.where(stress,0,np.nan),index=P.index).ffill().groupby((stress|stress.shift(1).isna()).cumsum()).cumcount()
# event-window recovery score only in the five sessions following an onset, then average
# recent event observations; negative relative performance is retained deliberately.
event_score=onset_rel.div(ownvol).where(age.between(1,5))
base=cs(event_score.rolling(20,min_periods=2).mean()).shift(1)
def residual(d):
 y=base.loc[d];X=pd.DataFrame({n:g.loc[d].reindex(A).fillna(0.) for n,g in S.items()},index=A);q=pd.concat([y.rename('y'),X],axis=1).dropna()
 if len(q)<8 or y.abs().sum()==0:return y*np.nan
 yy=q.y.to_numpy(float);xx=q.drop(columns='y').to_numpy(float);xx=(xx-xx.mean(0))/np.where(xx.std(0)>1e-12,xx.std(0),1);b=np.linalg.solve(xx.T@xx+5*np.eye(xx.shape[1]),xx.T@yy);o=pd.Series(np.nan,index=A);o.loc[q.index]=yy-xx@b;return o
cand=cs(pd.DataFrame({d:residual(d) for d in base.index}).T);fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR library_orthogonal_discrete_systemic_stress_onset_recovery_speed_5_20 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S),'ONSETS',int(stress.sum()))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in(1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
