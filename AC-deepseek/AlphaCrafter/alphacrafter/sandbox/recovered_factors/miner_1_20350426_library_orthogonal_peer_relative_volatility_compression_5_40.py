"""One candidate: library-orthogonal peer relative volatility compression transition (5/40)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']; r=P.pct_change()
# Asset-specific recent volatility compression relative to its own baseline and contemporaneous peer compression.
vol5=r.rolling(5,min_periods=4).std(); vol40=r.rolling(40,min_periods=30).std()
compression=1-vol5.div(vol40.replace(0,np.nan))
peer=compression.median(axis=1)
# Rank only the relative component: a persistent quieting versus peers may precede a differentiated continuation/repricing.
base=cs(compression.sub(peer,axis=0)).shift(1)
def residual(d):
 y=base.loc[d]; X=pd.DataFrame({n:g.loc[d].reindex(A).fillna(0.) for n,g in S.items()},index=A);q=pd.concat([y.rename('y'),X],axis=1).dropna()
 if len(q)<8 or y.abs().sum()==0:return y*np.nan
 yy=q.y.to_numpy(float);xx=q.drop(columns='y').to_numpy(float);xx=(xx-xx.mean(0))/np.where(xx.std(0)>1e-12,xx.std(0),1);yy=yy-yy.mean();b=np.linalg.solve(xx.T@xx+5*np.eye(xx.shape[1]),xx.T@yy);out=pd.Series(np.nan,index=A);out.loc[q.index]=yy-xx@b;return out
cand=cs(pd.DataFrame({d:residual(d) for d in base.index}).T);fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];vals=[];bread=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);bread.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'breadth':round(float(np.mean(bread)),3),'min_breadth':int(min(bread))}
print('FACTOR library_orthogonal_peer_relative_volatility_compression_transition_5_40 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
