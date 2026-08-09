"""One candidate: peer-floor drawdown-normalized systemic-weakness downside persistence (10/60)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']; r=P.pct_change()
peer=r.median(axis=1); pm=peer.rolling(60,min_periods=40).mean(); ps=peer.rolling(60,min_periods=40).std()
severity=1/(1+np.exp(-((pm-peer)/ps.replace(0,np.nan)).clip(-3,3)))
rel=r.sub(peer,axis=0); burden=(-rel.clip(upper=0)).mul(severity**2,axis=0)
short=burden.rolling(10,min_periods=6).mean(); long=burden.rolling(60,min_periods=40).mean()
# A 25% contemporaneous peer drawdown floor retains relative drawdown scaling but avoids discarding recovered assets.
dd=(1-P/P.rolling(40,min_periods=25).max()).rolling(10,min_periods=6).mean()
peer_floor=dd.median(axis=1).mul(.25)
den=dd.add(peer_floor,axis=0).replace(0,np.nan)
base=cs(short.div(long.replace(0,np.nan)).div(den)).shift(1)
def residual(d):
 y=base.loc[d];X=pd.DataFrame({n:g.loc[d].reindex(A).fillna(0.) for n,g in S.items()},index=A);q=pd.concat([y.rename('y'),X],axis=1).dropna()
 if len(q)<8 or y.abs().sum()==0:return y*np.nan
 yy=q.y.to_numpy(float)-q.y.mean();xx=q.drop(columns='y').to_numpy(float);xx=(xx-xx.mean(0))/np.where(xx.std(0)>1e-12,xx.std(0),1)
 b=np.linalg.solve(xx.T@xx+5*np.eye(xx.shape[1]),xx.T@yy);o=pd.Series(np.nan,index=A);o.loc[q.index]=yy-xx@b;return o
cand=cs(pd.DataFrame({d:residual(d) for d in base.index}).T);fw={h:P.shift(-h)/P-1 for h in(1,5,10,20)}
def stat(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];];v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),6),'breadth':round(np.mean(b),3),'minbreadth':min(b)}
print('FACTOR peer_floor_drawdown_normalized_severity_systemic_weakness_downside_persistence_10_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'LIBRARY',len(S))
print('CELLS',cand.notna().sum().sum(),'/',cand.size,'COVERAGE',round(cand.notna().stack().mean(),6),'TURNOVER',round(cand.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(cand.std(axis=1).mean(),6))
for h in fw:print('H',h,stat(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME20',n,stat(20,p))
mx=0;who='';ev=0;bad=[]
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8:bad.append(n);continue
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if not np.isfinite(rho):bad.append(n);continue
 if abs(rho)>mx:mx,who,ev=abs(rho),n,len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'INVALID_PAIRS',bad)
