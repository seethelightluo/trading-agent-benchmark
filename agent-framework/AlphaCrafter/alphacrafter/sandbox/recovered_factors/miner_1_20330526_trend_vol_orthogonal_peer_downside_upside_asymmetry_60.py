"""One idea: trend-and-volatility-orthogonal peer downside/upside asymmetry (60d).
An asset's average peer-relative return on broad-negative sessions less its
average peer-relative return on broad-positive sessions.  It is continuous in
coverage (rather than an event gate), residualized cross-sectionally each day
against 20d trend and idiosyncratic volatility, and lagged one session.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
m=r.median(axis=1); rel=r.sub(m,axis=0)
neg=rel.where(m<0,axis=0).rolling(60,min_periods=15).mean()
pos=rel.where(m>=0,axis=0).rolling(60,min_periods=15).mean()
raw=neg-pos
trend=P.pct_change(20)/r.rolling(20,min_periods=15).std()
idvol=rel.rolling(20,min_periods=15).std()
def orth(y,x1,x2):
 out=pd.DataFrame(np.nan,index=y.index,columns=y.columns)
 for d in y.index:
  q=pd.concat([y.loc[d],x1.loc[d],x2.loc[d]],axis=1).dropna()
  if len(q)<8: continue
  X=np.column_stack([np.ones(len(q)),q.iloc[:,1],q.iloc[:,2]])
  try: out.loc[d,q.index]=q.iloc[:,0]-X@np.linalg.lstsq(X,q.iloc[:,0],rcond=None)[0]
  except np.linalg.LinAlgError: pass
 return out
cand=cs(orth(raw,trend,idvol)).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1];]; y=fw[h].reindex(x.index); v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 if len(v)<2:return {'dates':len(v)}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR trend_vol_orthogonal_peer_downside_upside_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6),'NEG_SESSION_FRACTION',round(float((m<0).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME20',n,st(20,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 if len(q)<8:continue
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
